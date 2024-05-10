#!/mpcdf/soft/SLE_15/packages/x86_64/julia/1.10.1/bin/julia -t 13
# Standard output and error:
#SBATCH -o /ptmp/aenge/slang/data/derivatives/code/logs/julia.%j
#SBATCH -e /ptmp/aenge/slang/data/derivatives/code/logs/julia.%j
# Initial working directory:
#SBATCH -D /ptmp/aenge/slang/data/derivatives/code
# Job Name:
#SBATCH -J julia
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#
# Memory per node:
#SBATCH --mem 256G
#
# Wall clock limit:
#SBATCH --time=24:00:00

using CSV
using DataFrames
using Dates
using Glob
using MixedModels
using NIfTI
using OrderedCollections
using Pkg
using Printf
using StatsBase
using Suppressor
using TickTock

"Main function to compute group linear mixed models for contrasts of interest."
function main()

    pybids_dir = "/ptmp/aenge/slang/data/derivatives/pybids"
    beta_dir = "/ptmp/aenge/slang/data/derivatives/nilearn_vol"
    output_dir = "/ptmp/aenge/slang/data/derivatives/julia_vol"
    task = "language"
    space = "MNI152NLin2009cAsym"
    contrasts = [
        "audios_noise",
        "audios_pseudo",
        "audios_pseudo_minus_noise",
        "audios_words",
        "audios_words_minus_noise",
        "audios_words_minus_pseudo",
        "images_minus_audios",
        "images_noise",
        "images_pseudo",
        "images_pseudo_minus_noise",
        "images_words",
        "images_words_minus_noise",
        "images_words_minus_pseudo"]

    data_df_file = joinpath(pybids_dir, "task-$(task)_sessions.tsv")
    data_df = CSV.read(data_df_file, DataFrame)
    data_df.session = lpad.(data_df.session, 2, "0")
    data_df.ms = Dates.value.(data_df.acq_time - minimum(data_df.acq_time))
    data_df.days = Int.(round.(data_df.ms ./ (1000 * 60 * 60 * 24)))
    data_df.months = round.(data_df.days ./ 30.437, digits=3)
    data_df.months2 = data_df.months .^ 2
    data_df = filter(row -> row.fd_num_greater_0_5 <= (420 * 0.25), data_df)
    data_df = select(data_df, [:subject, :session, :months, :months2])
    data_df = filter(row -> occursin(r"^SA", row.subject), data_df)

    f0 = @formula(beta ~ 1 + (months + months2 | subject))
    f1 = @formula(beta ~ months + (months + months2 | subject))
    f2 = @formula(beta ~ months + months2 + (months + months2 | subject))

    if !isdir(output_dir)
        mkdir(output_dir)
    end

    resids_dir = joinpath(output_dir, "residuals")
    if !isdir(resids_dir)
        mkdir(resids_dir)
    end

    mask_file = string(
        beta_dir, "/task-", task, "_space-", space, "_desc-brain_mask.nii.gz")
    mask = niread(mask_file)
    voxel_ixs = findall(mask .== 1.0)
    n_voxels = length(voxel_ixs)

    Threads.@threads for contrast in contrasts

        beta_files = string.(
            beta_dir, "/sub-", data_df.subject, "/ses-", data_df.session,
            "/func/sub-", data_df.subject, "_ses-", data_df.session, "_task-",
            task, "_space-", space, "_desc-", contrast, "_effect_size.nii.gz")
        betas = niread.(beta_files)
        header = betas[1].header
        betas = cat(betas..., dims=4)

        res = OrderedDict(
            "x" => zeros(Int, n_voxels),
            "y" => zeros(Int, n_voxels),
            "z" => zeros(Int, n_voxels),
            "m0_b0" => zeros(n_voxels),
            "m0_z0" => zeros(n_voxels),
            "m0_p0" => zeros(n_voxels),
            "m1_b0" => zeros(n_voxels),
            "m1_b1" => zeros(n_voxels),
            "m1_z0" => zeros(n_voxels),
            "m1_z1" => zeros(n_voxels),
            "m1_p0" => zeros(n_voxels),
            "m1_p1" => zeros(n_voxels),
            "m1_m0_chisq" => zeros(n_voxels),
            "m1_m0_dof" => zeros(n_voxels),
            "m1_m0_pchisq" => zeros(n_voxels),
            "m2_b0" => zeros(n_voxels),
            "m2_b1" => zeros(n_voxels),
            "m2_b2" => zeros(n_voxels),
            "m2_z0" => zeros(n_voxels),
            "m2_z1" => zeros(n_voxels),
            "m2_z2" => zeros(n_voxels),
            "m2_p0" => zeros(n_voxels),
            "m2_p1" => zeros(n_voxels),
            "m2_p2" => zeros(n_voxels),
            "m2_m0_chisq" => zeros(n_voxels),
            "m2_m0_dof" => zeros(n_voxels),
            "m2_m0_pchisq" => zeros(n_voxels),
            "m2_m1_chisq" => zeros(n_voxels),
            "m2_m1_dof" => zeros(n_voxels),
            "m2_m1_pchisq" => zeros(n_voxels))

        resids = OrderedDict(
            "m0" => zeros(size(betas)),
            "m1" => zeros(size(betas)),
            "m2" => zeros(size(betas)))

        tick()
        for (ix, (x, y, z)) in enumerate(Tuple.(voxel_ixs))
            # println("Processing voxel ($x, $y, $z) ($ix/$n_voxels)")

            res["x"][ix] = x
            res["y"][ix] = y
            res["z"][ix] = z

            data_df.beta = betas[x, y, z, :]

            if length(unique(data_df.beta)) == 1
                value = only(unique(data_df.beta))
                println("Skipping voxel ($x, $y, $z) because data is constant ($value)")
                continue
            end

            begin
            #@suppress begin

                m0 = fit(MixedModel, f0, data_df)

                res["m0_b0"][ix] = only(m0.beta)
                res["m0_z0"][ix] = only(m0.beta) ./ only(m0.stderror)
                res["m0_p0"][ix] = only(m0.pvalues)

                resids["m0"][x, y, z, :] = StatsBase.residuals(m0)

                m1 = fit(MixedModel, f1, data_df)

                res["m1_b0"][ix],
                res["m1_b1"][ix] = m1.beta
                res["m1_z0"][ix],
                res["m1_z1"][ix] = m1.beta ./ m1.stderror
                res["m1_p0"][ix],
                res["m1_p1"][ix] = m1.pvalues

                resids["m1"][x, y, z, :] = StatsBase.residuals(m1)

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

                resids["m2"][x, y, z, :] = StatsBase.residuals(m2)

                res["m2_m0_chisq"][ix],
                res["m2_m0_dof"][ix],
                res["m2_m0_pchisq"][ix] = get_lrtest_stats(m2, m0)

                res["m2_m1_chisq"][ix],
                res["m2_m1_dof"][ix],
                res["m2_m1_pchisq"][ix] = get_lrtest_stats(m2, m1)

            end
        end
        tock()

        res_df = DataFrame(res)
        res_df = round_float_columns(res_df)

        res_df_filename = "sub-group_task-$(task)_space-$(space)_desc-$(contrast)_lmm.tsv"
        res_df_file = joinpath(output_dir, res_df_filename)
        CSV.write(res_df_file, res_df, delim='\t', missingstring="NaN")

        for (model, array) in resids
            # println("Writing residuals for model $model")
            vol = NIfTI.NIVolume(array)
            vol.header = header
            vol_filename = "sub-group_task-$(task)_space-$(space)_desc-$(contrast)_$(model)_residuals.nii.gz"
            vol_file = joinpath(resids_dir, vol_filename)
            niwrite(vol_file, vol)
        end
    end
end

"Computes model comparison (likelihood ratio test) statistics for two models."
function get_lrtest_stats(m2, m1)
    if m2.objective < m1.objective
        lrt = MixedModels.likelihoodratiotest(m1, m2)
        chisq = lrt.deviance[1] - lrt.deviance[2]
        dof = lrt.dof[2] - lrt.dof[1]
        pchisq = lrt.pvalues[1]
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
