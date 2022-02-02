#!/usr/bin/env bash
## Copyright (C) 2017, Oleksandr Kucherenko
## Last revisit: 2017-09-29

# For help:
#   ./versionip.sh --help

# For developer / references:
#  https://ryanstutorials.net/bash-scripting-tutorial/bash-functions.php
#  http://tldp.org/LDP/abs/html/comparison-ops.html
#  https://misc.flogisoft.com/bash/tip_colors_and_formatting

## display help
function help() {
  echo 'usage: ./version-up.sh [-r|--release] [-a|--alpha] [-b|--beta] [-c|--release-candidate]'
  echo '                      [-m|--major] [-i|--minor] [-p|--patch] [-e|--revision] [-g|--git-revision]'
  echo '                      [--stay] [--default] [--help]'
  echo ''
  echo 'Switches:'
  echo '  --release           switch stage to release, no suffix, -r'
  echo '  --alpha             switch stage to alpha, -a'
  echo '  --beta              switch stage to beta, -b'
  echo '  --release-candidate switch stage to rc, -c'
  echo '  --major             increment MAJOR version part, -m'
  echo '  --minor             increment MINOR version part, -i'
  echo '  --patch             increment PATCH version part, -p'
  echo ''
  echo 'Version: MAJOR.MINOR.PATCH'
  echo ''
  echo 'Reference:'
  echo '  http://semver.org/'
  echo ''
  echo 'Versions priority:'
  echo '  1.0.0-alpha < 1.0.0-beta < 1.0.0-rc < 1.0.0'
  exit 0
}

## parse last found tag, extract it PARTS
function parse_last() {
  local position=$(($1 - 1))

  # two parts found only
  local SUBS=(${PARTS[$position]//-/ })
  #echo ${SUBS[@]}, size: ${#SUBS}

  # found NUMBER
  PARTS[$position]=${SUBS[0]}
  #echo ${PARTS[@]}

  # found SUFFIX
  if [[ ${#SUBS} -ge 1 ]]; then
    PARTS[4]=${SUBS[1],,} #lowercase
    #echo ${PARTS[@]}, ${SUBS[@]}
  fi
}

## increment PATCH part, reset all other lower PARTS, don't touch STAGE
function increment_patch() {
  PARTS[2]=$((PARTS[2] + 1))
  PARTS[3]=0
}

## increment MINOR part, reset all other lower PARTS, don't touch STAGE
function increment_minor() {
  PARTS[1]=$((PARTS[1] + 1))
  PARTS[2]=0
  PARTS[3]=0
}

## increment MAJOR part, reset all other lower PARTS, don't touch STAGE
function increment_major() {
  PARTS[0]=$((PARTS[0] + 1))
  PARTS[1]=0
  PARTS[2]=0
  PARTS[3]=0
}

## compose version from PARTS
function compose() {
  MAJOR="${PARTS[0]}"
  MINOR=".${PARTS[1]}"
  PATCH=".${PARTS[2]}"

  echo "${MAJOR}${MINOR}${PATCH}" #full format
}

# parse input parameters
for i in "$@"; do
  key="$i"
  case $key in
  -v | --version) # tag prefix #fixme: will work only if placed as first argument
    VERSION=$2
    PARTS=(${VERSION//./ })
    parse_last ${#PARTS[@]} # array size as argument
    shift 2
    ;;
  -p | --patch) # increment of PATCH
    increment_patch
    ;;
  -i | --minor) # increment of MINOR by default
    increment_minor
    ;;
  -m | --major) # increment of MAJOR
    increment_major
    ;;
  esac
  shift
done

echo -e "$(compose)"
