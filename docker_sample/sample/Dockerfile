ARG UBUNTU_VER="18.04"
ARG CUDA_VER="10.2"
ARG CUDNN_VER="7"
FROM nvidia/cuda:${CUDA_VER}-cudnn${CUDNN_VER}-devel-ubuntu${UBUNTU_VER}

ARG UBUNTU_VER
ARG CUDA_VER
ARG CUDNN_VER
ARG PYTHON_VER="3.8"
ARG USER_NAME="crsprtvision"
ARG PW="0000"
ARG CONDA_PATH=/home/$USER_NAME/miniconda

ENV DEBIAN_FRONTEND noninteractive
ENV LIBRARY_PATH=$CONDA_PATH/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs
ENV LD_LIBRARY_PATH=$CONDA_PATH/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs
ENV PKG_CONFIG_PATH=$PKG_CONFIG_PATH:$CONDA_PATH/lib/pkgconfig

# Install packages from apt
RUN sed -i "s/archive.ubuntu/mirror.kakao/g" /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cuda-driver-dev-$CUDA_PKG_VERSION \
    cuda-misc-headers-$CUDA_PKG_VERSION \
    software-properties-common \
    libmount1 \
    sudo \
    git \
    tmux \
    vifm \
    curl \
    locales-all \
    xauth \
    unzip \
 && add-apt-repository -y ppa:neovim-ppa/stable \
 && apt-get update \
 && apt-get install -y --no-install-recommends neovim \
 && apt-get clean \
 && rm -rf \
    /tmp/* \
    /var/tmp/* \
    /var/lib/apt/lists/*

# Install video-codec-sdk
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},video
RUN VIDEOSDK_DOWNLOAD_SUM=389d5e73b36881b06ca00ea86f0e9c0c312c1646166b96669e8b51324943e213 \
 && curl -fsSL -O \
    https://developer.download.nvidia.com/compute/redist/VideoCodec/v8.2/NvCodec.zip  \
 && echo "$VIDEOSDK_DOWNLOAD_SUM  NvCodec.zip" | sha256sum -c - \
 && unzip -j NvCodec.zip \
    NvCodec/NvDecoder/cuviddec.h \
    NvCodec/NvDecoder/nvcuvid.h \
    NvCodec/NvEncoder/nvEncodeAPI.h \
    -d /usr/local/cuda/include \
 && unzip -j NvCodec.zip \
    NvCodec/Lib/linux/stubs/x86_64/libnvcuvid.so \
    NvCodec/Lib/linux/stubs/x86_64/libnvidia-encode.so \
    -d /usr/local/cuda/lib64/stubs \
 && rm NvCodec.zip

# Add non-root user
RUN useradd -ms /bin/bash ${USER_NAME} \
 && echo ${USER_NAME}:${PW} | chpasswd \
 && usermod -aG sudo ${USER_NAME}
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}/

# Install miniconda
ENV PATH=/home/${USER_NAME}/miniconda/bin:$PATH
RUN curl -so /home/${USER_NAME}/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /home/${USER_NAME}/miniconda.sh \
 && /home/${USER_NAME}/miniconda.sh -b -p /home/${USER_NAME}/miniconda \
 && rm /home/${USER_NAME}/miniconda.sh \
 && conda update -y conda && conda install -y conda-build \
 && conda install -y -c anaconda python=${PYTHON_VER} \
 && conda install -y -c pytorch \
    pytorch=1.6 torchvision=0.7 \
 && conda install -y -c conda-forge \
    pkg-config \
    meson \
    ninja \
    cmake \
    bison \
    flex \
    matplotlib \
    pycocotools \
    PySide2 \
    jupyter \
    python-language-server \
    pynvim \
    ffmpeg \
    eigen \
    harfbuzz \
    liblapacke \
    libwebp \
    gobject-introspection

# Build Gstreamer from source
RUN git clone https://gitlab.freedesktop.org/gstreamer/gst-build.git \
 && cd gst-build \
 && git checkout 1.18 \
 && meson build/ \
    --prefix=$CONDA_PATH \
    --libdir=lib \
    -Dtests=disabled \
    -Dexamples=disabled \
    -Dbenchmarks=disabled \
 && ninja -C build \
 && ninja -C build install \
 && rm -rf /home/$USER_NAME/gst-build

# Build Opencv from source
RUN git clone https://github.com/opencv/opencv \
 && git clone https://github.com/opencv/opencv_contrib \
 && cd opencv \
 && mkdir build \
 && cd build \
 && cmake \
    -DCMAKE_BUILD_TYPE="Release" \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_EXAMPLES=0 \
    -DBUILD_TESTS=0 \
    -DBUILD_DOCS=0 \
    -DBUILD_PERF_TESTS=0 \
    -DINSTALL_C_EXAMPLES=0 \
    -DINSTALL_PYTHON_EXAMPLES=0 \
    -DINSTALL_ANDROID_EXAMPLES=0 \
    -DINSTALL_BIN_DEXAMPLES=0 \
    -DENABLE_PYLINT=1 \
    -DENABLE_FLAKE8=1 \
    -DCMAKE_PREFIX_PATH=$CONDA_PATH \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PATH \
    -DCMAKE_INSTALL_LIBDIR="lib" \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DCUDA_FAST_MATH=ON \
    -DENABLE_FAST_MATH=ON \
    -DWITH_LAPACK=1 \
    -DWITH_EIGEN=1 \
    -DWITH_TIFF=1 \
    -DWITH_WEBP=1 \
    -DWITH_OPENEXR=1 \
    -DWITH_JASPER=1 \
    -DWITH_OPENJPEG=1 \
    -DWITH_V4L=1 \
    -DWITH_OPENGL=1 \
    -DWITH_CUDA=1 \
    -DWITH_CUBLAS=1 \
    -DWITH_CUDNN=1 \
    -DWITH_OPENCL=0 \
    -DWITH_OPENCLAMDFFT=0 \
    -DWITH_OPENCLAMDBLAS=0 \
    -DWITH_OPENCL_D3D11_NV=0 \
    -DWITH_1394=0 \
    -DWITH_OPENNI=0 \
    -DWITH_FFMPEG=1 \
    -DWITH_GSTREAMER=1 \
    -DWITH_DSHOW=0 \
    -DWITH_TENGINE=0 \
    -DWITH_MATLAB=0 \
    -DWITH_VTK=0 \
    -DWITH_GTK=0 \
    -DWITH_QT=5 \
    -DWITH_GPHOTO2=0 \
    -DOPENCV_EXTRA_MODULES_PATH="/home/$USER_NAME/opencv_contrib/modules" \
    -DCMAKE_SKIP_RPATH:bool=ON \
    -DBUILD_ZLIB=0 \
    -DBUILD_TIFF=0 \
    -DBUILD_PNG=0 \
    -DBUILD_OPENEXR=0 \
    -DBUILD_JASPER=0 \
    -DBUILD_JPEG=0 \
    -DBUILD_opencv_alphamat=1 \
    -DBUILD_opencv_bioinspired=1 \
    -DBUILD_opencv_core=1 \
    -DBUILD_opencv_cudaarithm=1 \
    -DBUILD_opencv_cudaimgproc=1 \
    -DBUILD_opencv_cudev=1 \
    -DBUILD_opencv_cudabgsegm=1 \
    -DBUILD_opencv_cudacodec=1 \
    -DBUILD_opencv_cudafeature2d=1 \
    -DBUILD_opencv_cudafilters=1 \
    -DBUILD_opencv_cudalegacy=1 \
    -DBUILD_opencv_cudaoptflow=1 \
    -DBUILD_opencv_cudaobjdetect=1 \
    -DBUILD_opencv_cudawarping=1 \
    -DBUILD_opencv_dnn=1 \
    -DBUILD_opencv_features2d=1 \
    -DBUILD_opencv_flann=1 \
    -DBUILD_opencv_fuzzy=1 \
    -DBUILD_opencv_gapi=1 \
    -DBUILD_opencv_hdf=1 \
    -DBUILD_opencv_hfs=1 \
    -DBUILD_opencv_highgui=1 \
    -DBUILD_opencv_img_hash=1 \
    -DBUILD_opencv_imgcodecs=1 \
    -DBUILD_opencv_imgproc=1 \
    -DBUILD_opencv_intensity_transform=1 \
    -DBUILD_opencv_line_descriptor=1 \
    -DBUILD_opencv_ml=1 \
    -DBUILD_opencv_objdetect=1 \
    -DBUILD_opencv_photo=1 \
    -DBUILD_opencv_python_bindings_generator=1 \
    -DBUILD_opencv_python2=0 \
    -DBUILD_opencv_python3=1 \
    -DBUILD_opencv_freetype=1 \
    -DBUILD_opencv_phase_unwrapping=1 \
    -DBUILD_opencv_plot=1 \
    -DBUILD_opencv_reg=1 \
    -DBUILD_opencv_shape=1 \
    -DBUILD_opencv_superres=1 \
    -DBUILD_opencv_ts=1 \
    -DBUILD_opencv_video=1 \
    -DBUILD_opencv_videoio=1 \
    -DZLIB_INCLUDE_DIR=$CONDA_PATH/include \
    -DZLIB_LIBRARY_RELEASE=$CONDA_PATH/lib/libz.so \
    -DJPEG_INCLUDE_DIR=$CONDA_PATH/include \
    -DTIFF_INCLUDE_DIR=$CONDA_PATH/include \
    -DPNG_PNG_INCLUDE_DIR=$CONDA_PATH/include \
    -DPYTHON2_EXECUTABLE="" \
    .. \
 && make -j8 && make install \
 && cd /home/$USER_NAME \
 && rm -rf /home/$USER_NAME/opencv /home/$USER_NAME/opencv_contrib \
 && conda clean -ya \
 && touch /home/${USER_NAME}/.Xauthority

CMD bash
