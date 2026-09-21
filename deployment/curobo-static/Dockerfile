# Isolated static simulation image. This image has no ROS launch entrypoint.
# Build context: offline wheel bundle and this directory's two Python files.
# build.py verifies the local base image ID before invoking Docker. This locally
# built image has no registry digest; an image config ID is not a FROM digest.
FROM rammp-curobo:jp6
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_INDEX=1
COPY verify_install.py /opt/rammp_static/verify_install.py
RUN python3 /opt/rammp_static/verify_install.py --before /opt/rammp_static/before.json
# setuptools_scm is already present here; the original source build produced
# 0.0.0 metadata before its runtime dependencies were installed. Rebuild from
# the exact unchanged checkout; never rewrite __version__ or dist-info by hand.
RUN python3 -m pip install --no-index --no-deps --no-build-isolation /opt/curobo
COPY wheels /opt/rammp_static/wheels
COPY wheels.lock /opt/rammp_static/wheels.lock
RUN python3 -m pip install --no-index --no-deps --require-hashes \
    --find-links /opt/rammp_static/wheels -r /opt/rammp_static/wheels.lock
RUN python3 /opt/rammp_static/verify_install.py --after /opt/rammp_static/after.json
ENTRYPOINT ["python3"]
CMD ["-c", "print('Static simulation image ready; no robot transport or default launch.')"]
