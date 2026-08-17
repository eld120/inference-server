ARG UBUNTU_VERSION=24.04
ARG SD_CPP_REPO=https://github.com/leejet/stable-diffusion.cpp.git
ARG SD_CPP_COMMIT=master
ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=master

FROM ubuntu:$UBUNTU_VERSION AS build
ARG SD_CPP_REPO
ARG SD_CPP_COMMIT

RUN apt update && apt install -y git build-essential cmake wget xz-utils curl libssl-dev \
    libvulkan-dev glslc spirv-headers libgomp1

WORKDIR /src

RUN git clone --recursive "${SD_CPP_REPO}" stable-diffusion.cpp \
    && cd stable-diffusion.cpp \
    && git checkout "${SD_CPP_COMMIT}"

WORKDIR /src/stable-diffusion.cpp

RUN cmake -B build \
        -DSD_VULKAN=ON \
        -DSD_BUILD_SERVER=ON \
        -DSD_BUILD_EXAMPLES=ON \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/bin /app/lib \
    && find build -name "*.so*" -exec cp -P {} /app/lib \; 2>/dev/null || true \
    && cp build/bin/sd-server /app/bin/ 2>/dev/null || cp build/bin/*server* /app/bin/ 2>/dev/null || cp build/bin/* /app/bin/

### Runtime image
FROM ubuntu:$UBUNTU_VERSION AS base

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=https://github.com/leejet/stable-diffusion.cpp
ARG IMAGE_SOURCE=https://github.com/leejet/stable-diffusion.cpp
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="stable-diffusion.cpp" \
      org.opencontainers.image.description="Diffusion model inference in C/C++ with Vulkan" \
      org.opencontainers.image.url=$IMAGE_URL \
      org.opencontainers.image.source=$IMAGE_SOURCE

RUN apt-get update \
    && apt-get install -y libgomp1 curl libvulkan1 vulkan-tools \
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
