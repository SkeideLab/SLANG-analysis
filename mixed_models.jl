using CSV
using DataFrames
using Dates
using Glob
using MixedModels
using OrderedCollections
using Pkg
using Printf
using TickTock

ENV["PYTHON"] = "/Users/alexander/mambaforge/envs/slang/bin/python"
Pkg.build("PyCall")
using PyCall
surface = pyimport("nilearn.surface")

"Main function to compute group linear mixed models for contrasts of interest."
function main()

    pybids_dir = "/Users/alexander/Research/slang/data/derivatives/pybids"
    bayesfmri_dir = "/Users/alexander/Research/slang/data/derivatives/bayesfmri"
    output_dir = "/Users/alexander/Research/slang/data/derivatives/julia"
    task = "language"
    run = "1"
    space = "fsaverage5"
    contrasts = OrderedDict(
        "audios-noise" => (("audios_noise",), ()),
        "audios-pseudo" => (("audios_pseudo",), ()),
        "audios-pseudo-minus-noise" => (("audios_pseudo",), ("audios_noise",)),
        "audios-words" => (("audios_words",), ()),
        "audios-words-minus-noise" => (("audios_words",), ("audios_noise",)),
        "audios-words-minus-pseudo" => (("audios_words",), ("audios_pseudo",)))
    contrasts = OrderedDict(
        "audios-minus-images" =>
            (("audios_noise", "audios_pseudo", "audios_words"),
                ("images_noise", "images_pseudo", "images_words")))

    data_df_file = joinpath(pybids_dir, "task-$(task)_sessions.tsv")
    data_df = CSV.read(data_df_file, DataFrame)
    data_df.session = lpad.(data_df.session, 2, "0")
    data_df.ms = Dates.value.(data_df.acq_time - minimum(data_df.acq_time))
    data_df.days = Int.(round.(data_df.ms ./ (1000 * 60 * 60 * 24)))
    data_df.months = round.(data_df.days ./ 30.437, digits=3)
    data_df.months2 = data_df.months .^ 2
    data_df = select(data_df, [:subject, :session, :months, :months2])
    data_df = filter(row -> occursin(r"^SA", row.subject), data_df)

    f0 = @formula(beta ~ 1 + (months + months2 | subject))
    f1 = @formula(beta ~ months + (months + months2 | subject))
    f2 = @formula(beta ~ months + months2 + (months + months2 | subject))

    for (desc, contrast) in contrasts

        for hemi in ["L", "R"]

            conditions_plus = contrast[1]
            betas_plus = read_betas_conditions(
                bayesfmri_dir, data_df.subject, data_df.session, hemi,
                conditions_plus)

            conditions_minus = contrast[2]
            if length(conditions_minus) == 0
                betas_minus = zeros(size(betas_plus))
            else
                betas_minus = read_betas_conditions(
                    bayesfmri_dir, data_df.subject, data_df.session, hemi,
                    conditions_minus)
            end

            betas = betas_plus - betas_minus
            n_vertices = size(betas, 1)

            res = OrderedDict(
                "vertex" => 1:n_vertices,
                "m0_b0" => zeros(n_vertices),
                "m0_z0" => zeros(n_vertices),
                "m0_p0" => zeros(n_vertices),
                "m1_b0" => zeros(n_vertices),
                "m1_b1" => zeros(n_vertices),
                "m1_z0" => zeros(n_vertices),
                "m1_z1" => zeros(n_vertices),
                "m1_p0" => zeros(n_vertices),
                "m1_p1" => zeros(n_vertices),
                "m1_m0_chisq" => zeros(n_vertices),
                "m1_m0_dof" => zeros(n_vertices),
                "m1_m0_pchisq" => zeros(n_vertices),
                "m2_b0" => zeros(n_vertices),
                "m2_b1" => zeros(n_vertices),
                "m2_b2" => zeros(n_vertices),
                "m2_z0" => zeros(n_vertices),
                "m2_z1" => zeros(n_vertices),
                "m2_z2" => zeros(n_vertices),
                "m2_p0" => zeros(n_vertices),
                "m2_p1" => zeros(n_vertices),
                "m2_p2" => zeros(n_vertices),
                "m2_m0_chisq" => zeros(n_vertices),
                "m2_m0_dof" => zeros(n_vertices),
                "m2_m0_pchisq" => zeros(n_vertices),
                "m2_m1_chisq" => zeros(n_vertices),
                "m2_m1_dof" => zeros(n_vertices),
                "m2_m1_pchisq" => zeros(n_vertices))

            tick()
            for ix in 1:n_vertices

                data_df.beta = betas[ix, :]

                m0 = fit(MixedModel, f0, data_df)
                res["m0_b0"][ix] = only(m0.beta)
                res["m0_z0"][ix] = only(m0.beta) ./ only(m0.stderror)
                res["m0_p0"][ix] = only(m0.pvalues)

                m1 = fit(MixedModel, f1, data_df)
                res["m1_b0"][ix],
                res["m1_b1"][ix] = m1.beta
                res["m1_z0"][ix],
                res["m1_z1"][ix] = m1.beta ./ m1.stderror
                res["m1_p0"][ix],
                res["m1_p1"][ix] = m1.pvalues

                res["m1_m0_chisq"][ix],
                res["m1_m0_dof"][ix],
                res["m1_m0_pchisq"][ix] = get_lrtest_stats(m1, m0)

                m2 = fit(MixedModel, f2, data_df)
                res["m2_b0"][ix],
                res["m2_b1"][ix],
                res["m2_b2"][ix] = m2.beta
                res["m2_z0"][ix],
                res["m2_z1"][ix],
                res["m2_z2"][ix] = m2.beta ./ m2.stderror
                res["m2_p0"][ix],
                res["m2_p1"][ix],
                res["m2_p2"][ix] = m2.pvalues

                res["m2_m0_chisq"][ix],
                res["m2_m0_dof"][ix],
                res["m2_m0_pchisq"][ix] = get_lrtest_stats(m2, m0)

                res["m2_m1_chisq"][ix],
                res["m2_m1_dof"][ix],
                res["m2_m1_pchisq"][ix] = get_lrtest_stats(m2, m1)

            end
            tock()

            res_df = DataFrame(res)
            res_df = round_float_columns(res_df)

            res_df_filename = "sub-group_task-$(task)_hemi-$(hemi)_space-$(space)_desc-$(desc)_lmm.tsv"
            res_df_file = joinpath(output_dir, res_df_filename)
            CSV.write(res_df_file, res_df, delim='\t', missingstring="NaN")
        end
    end
end

"Reads and averages first-level beta maps for a set of conditions."
function read_betas_conditions(bayesfmri_dir, subjects, sessions, hemi, conditions)
    betas = []
    for condition in conditions
        betas_condition = read_betas.(bayesfmri_dir, subjects, sessions, hemi, condition)
        betas_condition = stack(betas_condition, dims=2)
        push!(betas, betas_condition)
    end
    betas = stack(betas, dims=3)
    betas = dropdims(sum(betas, dims=3), dims=3)
    betas = betas ./ length(conditions)
end

"Reads first-level beta maps for a single condition."
function read_betas(bayesfmri_dir, subject, session, hemi, condition)
    beta_dir = joinpath(
        bayesfmri_dir, "sub-$(subject)", "ses-$(session)", "func")
    beta_filename = "sub-$(subject)_ses-$(session)_task-$(task)_run-$(run)_hemi-$(hemi)_space-$(space)_desc-$(condition)_effect_size.gii"
    beta_file = joinpath(beta_dir, beta_filename)
    surface.load_surf_data(beta_file)
end

"Computes model comparison (likelihood ratio test) statistics for two models."
function get_lrtest_stats(m2, m1)
    if m2.objective < m1.objective
        lrt = MixedModels.lrtest(m1, m2)
        chisq = lrt.deviance[1] - lrt.deviance[2]
        dof = lrt.dof[2] - lrt.dof[1]
        pchisq = lrt.pval[2]
    else
        chisq = NaN
        dof = NaN
        pchisq = NaN
    end
    return chisq, dof, pchisq
end

"Rounds float columns in a DataFrame to a certain number of decimal places."
function round_float_columns(df, digits=5)
    format = Ref(Printf.Format("%.$(digits)f"))
    for col in names(df)
        if eltype(df[!, col]) <: AbstractFloat
            df[!, col] = Printf.format.(format, df[!, col])
        end
    end
    return df
end

main()

# macro Name(arg)
#     string(arg)
# end

# arrays = [b_intercept, b_months, z_intercept, z_months,
#           p_intercept, p_months, chisq_months, chisq_dof_months,
#           p_chisq_months]
# array_labels = ["b_intercept", "b_months", "z_intercept", "z_months",
#                 "p_intercept", "p_months", "chisq_months", "chisq_dof_months",
#                 "p_chisq_months"]

# if !isdir(output_dir)
#     mkdir(output_dir)
# end

# nibabel = pyimport("nibabel")
# for (array, label) in zip(arrays, array_labels)
#     gifti_data = nibabel.gifti.GiftiDataArray(array, datatype="float32")
#     gifti_image = nibabel.gifti.GiftiImage(darrays=[gifti_data])
#     gifti_filename = "task-$(task)_hemi-L_space-$(space)_desc-" * contrast * "_" * label * ".gii"
#     gifti_file = joinpath(output_dir, gifti_filename)
#     nibabel.save(gifti_image, gifti_file)
# end
