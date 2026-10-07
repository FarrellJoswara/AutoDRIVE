# Competition policy image: extend the official IROS Devkit image and add only
# the team's model/runtime. The official simulator remains a separate image.
FROM autodriveecosystem/autodrive_roboracer_api:2026-iros-compete

ENV PYTHONUNBUFFERED=1
ENV AICAR_MODEL_PATH=/models/policy.zip
WORKDIR /opt/aicar

# SB3/PyTorch are not included in the official Devkit image. Add only the
# policy runtime; in particular, retain the official ROS NumPy version.
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.2.2+cpu \
    && python3 -m pip install --no-cache-dir \
      numpy==1.22.2 gymnasium==0.29.1 cloudpickle==3.1.2 \
      pandas==1.5.3 matplotlib==3.6.3 tqdm==4.67.1 \
    && python3 -m pip install --no-cache-dir --no-deps stable-baselines3==2.3.2

COPY src /opt/aicar/src
COPY competition/iros2026/run_container.sh /opt/aicar/run_container.sh
RUN chmod 0755 /opt/aicar/run_container.sh

ENTRYPOINT ["/bin/bash", "/opt/aicar/run_container.sh"]
