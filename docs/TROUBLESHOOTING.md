# Troubleshooting — VA Share Quick Tips

| Issue | Fix |
| ----- | --- |
| **"Next" button stays disabled** | Ensure every required label has a value or N/A (when allowed). | 
| **"Cannot save" message** | Verify the reviewer has write permissions to the assignment folder; the client caches locally and retries automatically once access is restored. |
| **"Text hash mismatch" during import** | The corpus changed after assignment creation. Re-run the round generation with the canonical corpus. |
| **Assignment marked "Locked"** | Confirm no reviewer has the client open. If safe, delete the stale `.lock` file inside the assignment folder. |
