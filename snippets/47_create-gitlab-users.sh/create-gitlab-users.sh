#!/bin/bash

if [[ -z "$GITLAB_PAT" ]]; then
  echo "[!] GITLAB_PAT environment is not set"
  exit 1
fi

# Create group, user and assign user to group
group=( developer security operations compliance )
user=( louis joe alan robert )
level=( 30 30 40 10 )

for ((i=0; i<${#group[@]}; i++)); do
  echo "[+] Create user '${user[$i]}'"
  curl -s -o /dev/null -XPOST -H "Content-Type: application/json" \
    -d '{ "email": "'${user[$i]}'@'${GITLAB_HOST}'", "username": "'${user[$i]}'", "password": "'${GITLAB_PASS}'", "name": "'${user[$i]}'", "skip_reconfirmation": "true" }' \
    $GITLAB_HOST/api/v4/users?private_token=$GITLAB_PAT

  user_id=$(curl -s -XGET -H "Content-Type: application/json" \
    -H "PRIVATE-TOKEN: $GITLAB_PAT" \
    $GITLAB_HOST/api/v4/users?username=${user[$i]} | jq -r '(.[0]? // .) .id')

  echo "[+] Add user '${user[$i]}' to group '${group[$i]}'"
  curl -s -o /dev/null -XPOST \
    -d "user_id=${user_id}&access_level=${level[$i]}" \
    $GITLAB_HOST/api/v4/groups/${group[$i]}/members?private_token=$GITLAB_PAT
done