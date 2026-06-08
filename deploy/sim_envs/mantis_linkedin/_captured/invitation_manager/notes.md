# /mynetwork/invitation-manager/

## Layout

Single centre column 720px. Top: tabs Received | Sent | I don't know.

## Received tab

- Header row: "N pending invitations" + sort dropdown (Newest / Oldest).
- List rows. Each row: avatar 56px, name + headline + "Sent N days ago",
  "Ignore" outlined + "Accept" filled blue (right-aligned).

## Sent tab

- Header "N pending requests".
- Same row shape but actions are "Withdraw" outlined.

## Interactions

- Accept: writes connections.status='accepted' + audit_log row.
- Withdraw: removes row from sent + audit_log.
