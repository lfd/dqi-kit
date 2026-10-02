# Pinned by digest so that the base layer does not follow the floating tag.
FROM archlinux@sha256:70aace0bf67d14280ca54bc2c7ee15c5fff62684131b34ab38f99f730b643733

ENV TERM=xterm

# Every Arch package (SageMath, GAP, Python, numpy, ...) is resolved against one
# snapshot of the Arch Linux Archive, so the system layer is the same on every
# build. Bump the date to upgrade the whole system layer deliberately.
ARG ARCH_SNAPSHOT=2026/10/02

RUN echo "Server = https://archive.archlinux.org/repos/${ARCH_SNAPSHOT}/\$repo/os/\$arch" \
        > /etc/pacman.d/mirrorlist && \
    pacman -Sy --noconfirm archlinux-keyring && \
    pacman -Syu --noconfirm \
        python \
        python-pip \
        sagemath \
        gap-guava \
        gap-packages \
        base-devel \
        git \
        && pacman -Scc --noconfirm

WORKDIR /app

RUN python -m venv /app/venv --system-site-packages

ENV PATH="/app/venv/bin:$PATH"

# Python packages are pinned in requirements.txt; pip itself comes from the
# pinned python-pip package above.
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "examples.py"]
