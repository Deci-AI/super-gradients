#!/usr/bin/env bash

set -e

# setup default values
cfg_file="$HOME/.aws-sudo"

duration=3600

# parse command line
while [ "$#" -gt 0 ]; do
  case "$1" in
  -n)
    session_name="$2"
    shift 2
    ;;
  -c)
    command="$2"
    shift 2
    ;;
  -x)
    clear=1
    shift 1
    ;;
  -f)
    cfg_file="$2"
    shift 2
    ;;
  -p)
    profile="$2"
    shift 2
    ;;
  -d)
    duration="$2"
    shift 2
    ;;
  -h)
    cat 1>&2 <<EOF
$(basename "$0") [-n sess_name] [-c command] [-x] [-f cfg_file] [-p profile] argument
Request credentials via STS and prepare environment variables for the
AWS SDKs.  By default, generates Bourne-shell code to be eval'ed.
optional args:
  -n sess_name	Session name for STS
  -c command	Run a command as the role; passed to "sh -c"
  -x		Generate command to clean modified environment vars
  -f cfg_file	Override config file for defaults and aliases
  -p profile	Use a non-default AWS profile when calling STS
  -d duration   Session duration 12 hours default
positional args:
  argument:	Must be one of:
			full role ARN
			a configured alias name
			12-digit AWS account number
			the literal "clear" (equivalent to -x)
EOF
    exit
    ;;
  *)
    argument=$1
    shift 1
    ;;
  esac
done

# handle unset requests and exit
if [[ "$argument" == "clear" || "$clear" == "1" ]]; then
  echo "unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SECURITY_TOKEN AWS_SESSION_TOKEN"
  exit
fi

# if the arg doesn't look like an arn, check for aliases
if [[ "$argument" =~ arn:aws:iam::[0-9]{12}:role/ ]]; then
  role="$argument"
else
  if [ -r $cfg_file ]; then
    alias=$(grep "^alias $argument" $cfg_file 2>/dev/null | head -n 1)
    role=$(echo "$alias" | awk '{print $3}')

    # if no session name was specified, look for one in the alias
    session_name=${session_name:-$(echo "$alias" | awk '{print $4}')}
  fi
fi

# if argument is an aws account number, look for a default role name
# in the config.  If found, build the role arn using that default
if [[ -z "$role" && "$argument" =~ ^[0-9]{12}$ ]]; then
  def_role_name=$(grep "^default role " $cfg_file 2>/dev/null | awk '{print $3}' | head -n 1)
  if [ -n "$def_role_name" ]; then
    role="arn:aws:iam::${argument}:role/${def_role_name}"
  fi
fi

# if no session name was provided, try to find a default
if [ -z "$session_name" ]; then
  def_session_name=$(grep "^default session_name" $cfg_file 2>/dev/null | awk '{print $3}')
  session_name=${def_session_name:-aws_sudo}
fi

# if no source profile was provided, try to find a default
if [ -z "$profile" ]; then
  profile=$(grep "^default profile" $cfg_file 2>/dev/null | awk '{print $3}')
fi

# verify that a valid role arn was found or provided; awscli gives
# terrible error messages if you try to assume some non-arn junk
if ! [[ "$role" =~ arn:aws:iam::[0-9]{12}:role/ ]]; then
  echo "$argument is neither a role ARN nor a configured alias" 1>&2
  exit 1
fi

response=$(aws ${profile:+--profile $profile} \
  sts assume-role --output text \
  --role-arn "$role" \
  --role-session-name="$session_name" \
  --duration-seconds=$duration \
  --query Credentials)

if [ -n "$command" ]; then
  env \
    AWS_ACCESS_KEY_ID=$(echo $response | awk '{print $1}') \
    AWS_SECRET_ACCESS_KEY=$(echo $response | awk '{print $3}') \
    AWS_SESSION_TOKEN=$(echo $response | awk '{print $4}') \
    bash -c "$command"
else
  echo export \
    AWS_ACCESS_KEY_ID=$(echo $response | awk '{print $1}') \
    AWS_SECRET_ACCESS_KEY=$(echo $response | awk '{print $3}') \
    AWS_SESSION_TOKEN=$(echo $response | awk '{print $4}')
fi
