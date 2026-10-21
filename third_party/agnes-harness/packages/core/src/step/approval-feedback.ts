export const approvalRejectionText = (reason: string): string =>
  `${reason}. This tool call was not executed. Confirm the current state before dependent work; do not assume a refused modification took effect. Retrying still requires approval; request authorization or use currently permitted operations. Avoid repeating the same refused call while authorization is unchanged.`
