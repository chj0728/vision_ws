#!/usr/bin/env bash
set -Eeuo pipefail

# 将 run.sh 注册为 Supervisor 服务，实现开机自启动和异常退出后的自动拉起。
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
WORK_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
RUN_SCRIPT="$SCRIPT_DIR/run.sh"
PROGRAM_NAME="vision_stack"
SUPERVISOR_CONF="/etc/supervisor/conf.d/${PROGRAM_NAME}.conf"
SUPERVISOR_LOG_DIR="$WORK_DIR/logs"

log() {
	printf '[INFO] %s\n' "$*"
}

fail() {
	printf '[ERROR] %s\n' "$*" >&2
	exit 1
}

usage() {
	cat <<EOF
用法: $(basename "$0") [命令]

命令:
	deploy    安装 Supervisor、写入配置并启动视觉栈（默认）
	start     启动视觉栈
	stop      停止视觉栈
	restart   重启视觉栈
	status    查看视觉栈状态
	remove    停止视觉栈并删除 Supervisor 配置
	logs      持续查看视觉栈日志
	help      显示此帮助信息
EOF
}

# 未安装 Supervisor 时使用 apt 自动安装。
install_supervisor() {
	if command -v supervisord >/dev/null 2>&1 && command -v supervisorctl >/dev/null 2>&1; then
		return
	fi

	command -v apt-get >/dev/null 2>&1 || fail "未找到 apt-get，请手动安装 Supervisor。"
	log "正在安装 Supervisor..."
	sudo apt-get update
	sudo DEBIAN_FRONTEND=noninteractive apt-get install -y supervisor
}

# 启动 Supervisor 服务并设置为开机自启。
enable_supervisor() {
	if command -v systemctl >/dev/null 2>&1; then
		sudo systemctl enable --now supervisor
	else
		sudo service supervisor start
	fi
}

# 生成配置时使用调用者而非 root 运行 ROS 节点，避免工作区文件权限错乱。
write_config() {
	local run_user user_home ros_domain_id ros_discovery_range temp_conf
	run_user=${SUDO_USER:-$(id -un)}
	user_home=$(getent passwd "$run_user" | cut -d: -f6)
	[[ -n "$user_home" ]] || fail "无法确定用户目录：$run_user"

	# Supervisor 不会继承当前终端环境；DDS 配置必须与命令行保持一致。
	ros_domain_id=${ROS_DOMAIN_ID:-0}
	ros_discovery_range=${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}
	[[ "$ros_domain_id" =~ ^[0-9]+$ ]] || fail "ROS_DOMAIN_ID 必须是非负整数：$ros_domain_id"
	log "使用 ROS_DOMAIN_ID=$ros_domain_id，ROS_AUTOMATIC_DISCOVERY_RANGE=$ros_discovery_range"

	mkdir -p "$SUPERVISOR_LOG_DIR"
	temp_conf=$(mktemp)
	cat > "$temp_conf" <<EOF
[program:$PROGRAM_NAME]
directory=$WORK_DIR
command=$RUN_SCRIPT
user=$run_user
autostart=true
autorestart=true
startsecs=5
stopsignal=INT
stopasgroup=true
killasgroup=true
environment=HOME="$user_home",USER="$run_user",ROS_DOMAIN_ID="$ros_domain_id",ROS_AUTOMATIC_DISCOVERY_RANGE="$ros_discovery_range"
stdout_logfile=$SUPERVISOR_LOG_DIR/supervisor.log
stderr_logfile=$SUPERVISOR_LOG_DIR/supervisor.log
EOF

	sudo install -m 0644 "$temp_conf" "$SUPERVISOR_CONF"
	rm -f "$temp_conf"
	log "已写入 Supervisor 配置：$SUPERVISOR_CONF"
}

# 重新读取配置，配置首次出现时会由 update 自动添加并启动。
apply_config() {
	sudo supervisorctl reread
	sudo supervisorctl update
	sudo supervisorctl start "$PROGRAM_NAME" 2>/dev/null || true
	sudo supervisorctl status "$PROGRAM_NAME"
}

deploy() {
	[[ -f "$RUN_SCRIPT" ]] || fail "未找到运行脚本：$RUN_SCRIPT"
	[[ -x "$RUN_SCRIPT" ]] || fail "运行脚本不可执行，请执行：chmod +x $RUN_SCRIPT"

	install_supervisor
	enable_supervisor
	write_config
	apply_config
}

# 停止并注销视觉栈服务，仅删除本脚本生成的 Supervisor 配置。
remove() {
	if sudo test -f "$SUPERVISOR_CONF"; then
		sudo supervisorctl stop "$PROGRAM_NAME" 2>/dev/null || true
		sudo rm -f "$SUPERVISOR_CONF"
		sudo supervisorctl reread
		sudo supervisorctl update
		log "已移除视觉栈开机自启动配置。"
	else
		log "未找到视觉栈 Supervisor 配置，无需移除。"
	fi
}

main() {
	local command=${1:-deploy}
	command -v sudo >/dev/null 2>&1 || fail "需要 sudo 管理 Supervisor。"

	case "$command" in
		deploy)
			deploy
			;;
		start|stop|restart|status)
			sudo supervisorctl "$command" "$PROGRAM_NAME"
			;;
		remove)
			remove
			;;
		logs)
			tail -n 100 -f "$WORK_DIR/logs/start.launch.log" "$SUPERVISOR_LOG_DIR/supervisor.log"
			;;
		help|-h|--help)
			usage
			;;
		*)
			usage >&2
			fail "未知命令：$command"
			;;
	esac
}

main "$@"
