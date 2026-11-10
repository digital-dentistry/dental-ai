#!/bin/sh
# 컨테이너 시작 시 SSH 접속을 준비한 뒤 CMD(기본: sshd)나 `docker compose run`의 명령을 실행한다.
set -e

# 공개키: 마운트된 docker/authorized_keys를 sshd가 요구하는 권한(600)으로 복사 (Windows 마운트는 권한이 777로 보인다)
mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ -f /app/docker/authorized_keys ]; then
    tr -d '\r' < /app/docker/authorized_keys > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

# 호스트 키: 볼륨에 두어 이미지를 다시 빌드해도 바뀌지 않게 한다 (바뀌면 접속 측에서 경고가 뜬다)
[ -f /etc/ssh/host_keys/ssh_host_ed25519_key ] || ssh-keygen -q -t ed25519 -N '' -f /etc/ssh/host_keys/ssh_host_ed25519_key

# SSH 세션은 컨테이너 ENV를 물려받지 않으므로 /etc/environment(pam_env)로 넘긴다 (DDH_: dental_ai ddh-fetch 접속 정보)
printenv | grep -E '^(LANG|PYTHON[A-Z_]*|PIP_[A-Z_]*|MPLBACKEND|NVIDIA_[A-Z_]*|TF_[A-Z_]*|DDH_[A-Z0-9_]*)=' > /etc/environment || true

# `dental_ai` 명령: 소스는 이미지가 아니라 마운트로 들어오므로 시작할 때 editable로 설치한다 (오프라인, 몇 초).
# 실패해도 sshd는 떠야 하므로 set -e에 걸리지 않게 한다.
pip install -e /app --no-deps --no-build-isolation -q --root-user-action=ignore \
    || echo "entrypoint: dental_ai 설치 실패 -- 'dental_ai' 명령 없이 계속한다" >&2

mkdir -p /run/sshd
exec "$@"
