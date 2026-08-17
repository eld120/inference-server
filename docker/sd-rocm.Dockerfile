ARG UBUNTU_VERSION=24.04
ARG ROCM_VERSION=7.2.4
ARG SD_CPP_REPO=https://github.com/leejet/stable-diffusion.cpp.git
ARG SD_CPP_COMMIT=master

# Target the ROCm build image
ARG BASE_ROCM_DEV_CONTAINER=rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}-complete

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=master

### Build image
FROM ${BASE_ROCM_DEV_CONTAINER} AS build
ARG SD_CPP_REPO
ARG SD_CPP_COMMIT

ARG ROCM_DOCKER_ARCH='gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1150;gfx1200;gfx1201'
ENV AMDGPU_TARGETS=${ROCM_DOCKER_ARCH}

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    cmake \
    git \
    libssl-dev \
    curl \
    libgomp1

WORKDIR /src

RUN git clone --recursive "${SD_CPP_REPO}" stable-diffusion.cpp \
    && cd stable-diffusion.cpp \
    && git checkout "${SD_CPP_COMMIT}"

WORKDIR /src/stable-diffusion.cpp

RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
        -DSD_HIP=ON \
        -DSD_HIPBLAS=ON \
        -DAMDGPU_TARGETS="$ROCM_DOCKER_ARCH" \
        -DSD_BUILD_SERVER=ON \
        -DSD_BUILD_EXAMPLES=ON \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/bin /app/lib \
    && find build -name "*.so*" -exec cp -P {} /app/lib \; 2>/dev/null || true \
    && cp build/bin/sd-server /app/bin/ 2>/dev/null || cp build/bin/*server* /app/bin/ 2>/dev/null || cp build/bin/* /app/bin/

### Runtime image
FROM ${BASE_ROCM_DEV_CONTAINER} AS base

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=https://github.com/leejet/stable-diffusion.cpp
ARG IMAGE_SOURCE=https://github.com/leejet/stable-diffusion.cpp
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="stable-diffusion.cpp" \
      org.opencontainers.image.description="Diffusion and image/video model inference in C/C++" \
      org.opencontainers.image.url=$IMAGE_URL \
      org.opencontainers.image.source=$IMAGE_SOURCE

RUN apt-get update \
    && apt-get install -y libgomp1 curl \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

COPY --from=build /app/lib/ /usr/local/lib/
COPY --from=build /app/bin/ /app/

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
WORKDIR /app

ENTRYPOINT ["/app/sd-server"]
