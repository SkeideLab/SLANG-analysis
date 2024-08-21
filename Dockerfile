# Generated by Neurodocker and Reproenv.

FROM fedora:36
ENV PATH="/opt/afni-latest:$PATH" \
    AFNI_PLUGINPATH="/opt/afni-latest"
RUN yum install -y -q \
           R \
           cmake \
           curl \
           ed \
           gsl \
           libGLU \
           libXp \
           libXpm \
           libcurl-devel \
           libgomp \
           libjpeg-turbo-devel \
           libpng12 \
           mesa-dri-drivers \
           mesa-dri-drivers \
           mesa-libGLw \
           ncurses-compat-libs \
           netpbm-progs \
           openmotif \
           openssl-devel \
           python-is-python3 \
           python3-pip \
           tcsh \
           udunits2-devel \
           unzip \
           wget \
           which \
           which \
           xorg-x11-fonts-misc \
           xorg-x11-server-Xvfb \
    && yum clean all \
    && rm -rf /var/cache/yum/* \
    && gsl_path="$(find / -name 'libgsl.so.??' || printf '')" \
    && if [ -n "$gsl_path" ]; then \
         ln -sfv "$gsl_path" "$(dirname $gsl_path)/libgsl.so.0"; \
    fi \
    && ldconfig \
    && mkdir -p /opt/afni-latest \
    && echo "Downloading AFNI ..." \
    && curl -fL https://afni.nimh.nih.gov/pub/dist/tgz/linux_openmp_64.tgz \
    | tar -xz -C /opt/afni-latest --strip-components 1
ENV PYTHON_JULIAPKG_PROJECT="/opt/miniconda-latest/julia_env"
ENV CONDA_DIR="/opt/miniconda-latest" \
    PATH="/opt/miniconda-latest/bin:$PATH"
RUN yum install -y -q \
           bzip2 \
           curl \
    && yum clean all \
    && rm -rf /var/cache/yum/* \
    # Install dependencies.
    && export PATH="/opt/miniconda-latest/bin:$PATH" \
    && echo "Downloading Miniconda installer ..." \
    && conda_installer="/tmp/miniconda.sh" \
    && curl -fsSL -o "$conda_installer" https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash "$conda_installer" -b -p /opt/miniconda-latest \
    && rm -f "$conda_installer" \
    && conda update -yq -nbase conda \
    # Prefer packages in conda-forge
    && conda config --system --prepend channels conda-forge \
    # Packages in lower-priority channels not considered if a package with the same
    # name exists in a higher priority channel. Can dramatically speed up installations.
    # Conda recommends this as a default
    # https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html
    && conda config --set channel_priority strict \
    && conda config --system --set auto_update_conda false \
    && conda config --system --set show_channel_urls true \
    # Enable `conda activate`
    && conda init bash \
    && conda install -y  --name base \
           "matplotlib" \
           "nilearn" \
           "pybids" \
           "pyjuliacall" \
           "seaborn" \
    # Clean up
    && sync && conda clean --all --yes && sync \
    && rm -rf ~/.cache/pip/*
RUN python3 -c 'import juliapkg as jpkg; jpkg.add("InlineStrings", uuid="842dd82b-1e85-43dc-bf29-5d0ee9dffc48"); jpkg.add("MixedModels", uuid="ff71e718-51f3-5ec2-a782-8ffcbfa3c316"); jpkg.add("PythonCall", uuid="6099a3de-0909-46bc-b1f4-468b9a2dfc0d"); jpkg.add("StatsBase", uuid="2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"); jpkg.add("Suppressor", uuid="fd094767-a336-5f1f-9728-57cf17d0bbfb"); jpkg.add("Tables", uuid="bd369af6-aec1-5ad0-b16a-f7cc5008161c"); from juliacall import Main as jl; jl.seval("using Pkg; Pkg.instantiate(); Pkg.precompile()")'
COPY ["univariate.py", \
      "/opt/"]
ENTRYPOINT ["python3", "/opt/univariate.py"]

# Save specification to JSON.
RUN printf '{ \
  "pkg_manager": "yum", \
  "existing_users": [ \
    "root" \
  ], \
  "instructions": [ \
    { \
      "name": "from_", \
      "kwds": { \
        "base_image": "fedora:36" \
      } \
    }, \
    { \
      "name": "env", \
      "kwds": { \
        "PATH": "/opt/afni-latest:$PATH", \
        "AFNI_PLUGINPATH": "/opt/afni-latest" \
      } \
    }, \
    { \
      "name": "run", \
      "kwds": { \
        "command": "yum install -y -q \\\\\\n    R \\\\\\n    cmake \\\\\\n    curl \\\\\\n    ed \\\\\\n    gsl \\\\\\n    libGLU \\\\\\n    libXp \\\\\\n    libXpm \\\\\\n    libcurl-devel \\\\\\n    libgomp \\\\\\n    libjpeg-turbo-devel \\\\\\n    libpng12 \\\\\\n    mesa-dri-drivers \\\\\\n    mesa-dri-drivers \\\\\\n    mesa-libGLw \\\\\\n    ncurses-compat-libs \\\\\\n    netpbm-progs \\\\\\n    openmotif \\\\\\n    openssl-devel \\\\\\n    python-is-python3 \\\\\\n    python3-pip \\\\\\n    tcsh \\\\\\n    udunits2-devel \\\\\\n    unzip \\\\\\n    wget \\\\\\n    which \\\\\\n    which \\\\\\n    xorg-x11-fonts-misc \\\\\\n    xorg-x11-server-Xvfb\\nyum clean all\\nrm -rf /var/cache/yum/*\\n\\ngsl_path=\\"$\(find / -name '"'"'libgsl.so.??'"'"' || printf '"'"''"'"'\)\\"\\nif [ -n \\"$gsl_path\\" ]; then \\\\\\n  ln -sfv \\"$gsl_path\\" \\"$\(dirname $gsl_path\)/libgsl.so.0\\"; \\\\\\nfi\\nldconfig\\nmkdir -p /opt/afni-latest\\necho \\"Downloading AFNI ...\\"\\ncurl -fL https://afni.nimh.nih.gov/pub/dist/tgz/linux_openmp_64.tgz \\\\\\n| tar -xz -C /opt/afni-latest --strip-components 1" \
      } \
    }, \
    { \
      "name": "env", \
      "kwds": { \
        "PYTHON_JULIAPKG_PROJECT": "/opt/miniconda-latest/julia_env" \
      } \
    }, \
    { \
      "name": "env", \
      "kwds": { \
        "CONDA_DIR": "/opt/miniconda-latest", \
        "PATH": "/opt/miniconda-latest/bin:$PATH" \
      } \
    }, \
    { \
      "name": "run", \
      "kwds": { \
        "command": "yum install -y -q \\\\\\n    bzip2 \\\\\\n    curl\\nyum clean all\\nrm -rf /var/cache/yum/*\\n# Install dependencies.\\nexport PATH=\\"/opt/miniconda-latest/bin:$PATH\\"\\necho \\"Downloading Miniconda installer ...\\"\\nconda_installer=\\"/tmp/miniconda.sh\\"\\ncurl -fsSL -o \\"$conda_installer\\" https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh\\nbash \\"$conda_installer\\" -b -p /opt/miniconda-latest\\nrm -f \\"$conda_installer\\"\\nconda update -yq -nbase conda\\n# Prefer packages in conda-forge\\nconda config --system --prepend channels conda-forge\\n# Packages in lower-priority channels not considered if a package with the same\\n# name exists in a higher priority channel. Can dramatically speed up installations.\\n# Conda recommends this as a default\\n# https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html\\nconda config --set channel_priority strict\\nconda config --system --set auto_update_conda false\\nconda config --system --set show_channel_urls true\\n# Enable `conda activate`\\nconda init bash\\nconda install -y  --name base \\\\\\n    \\"matplotlib\\" \\\\\\n    \\"nilearn\\" \\\\\\n    \\"pybids\\" \\\\\\n    \\"pyjuliacall\\" \\\\\\n    \\"seaborn\\"\\n# Clean up\\nsync && conda clean --all --yes && sync\\nrm -rf ~/.cache/pip/*" \
      } \
    }, \
    { \
      "name": "run", \
      "kwds": { \
        "command": "python3 -c '"'"'import juliapkg as jpkg; jpkg.add\(\\"InlineStrings\\", uuid=\\"842dd82b-1e85-43dc-bf29-5d0ee9dffc48\\"\); jpkg.add\(\\"MixedModels\\", uuid=\\"ff71e718-51f3-5ec2-a782-8ffcbfa3c316\\"\); jpkg.add\(\\"PythonCall\\", uuid=\\"6099a3de-0909-46bc-b1f4-468b9a2dfc0d\\"\); jpkg.add\(\\"StatsBase\\", uuid=\\"2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91\\"\); jpkg.add\(\\"Suppressor\\", uuid=\\"fd094767-a336-5f1f-9728-57cf17d0bbfb\\"\); jpkg.add\(\\"Tables\\", uuid=\\"bd369af6-aec1-5ad0-b16a-f7cc5008161c\\"\); from juliacall import Main as jl; jl.seval\(\\"using Pkg; Pkg.instantiate\(\); Pkg.precompile\(\)\\"\)'"'"'" \
      } \
    }, \
    { \
      "name": "copy", \
      "kwds": { \
        "source": [ \
          "univariate.py", \
          "/opt/" \
        ], \
        "destination": "/opt/" \
      } \
    }, \
    { \
      "name": "entrypoint", \
      "kwds": { \
        "args": [ \
          "python3", \
          "/opt/univariate.py" \
        ] \
      } \
    } \
  ] \
}' > /.reproenv.json
# End saving to specification to JSON.
