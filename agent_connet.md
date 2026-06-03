# Agent Connection Guide

This document describes how agents should connect to and communicate with me (the user).

---

## Communication Protocol

### 1. Initial Connection

When connecting to me, agents should:

1. **Identify themselves** - State your name/role and purpose
2. **Confirm context** - Verify you understand the current working directory and project
3. **Ask for clarification** - If the task is unclear, ask questions before proceeding

Example:
```
Hello, I'm [Agent Name]. I see we're working on [Project Name].
You asked me to [task]. Let me confirm: [clarification question]
```

### 2. Task Execution

While working on tasks:

- **Report progress** - Give regular updates on what you're doing
- **Show your work** - Use tools to actually make changes, not just describe them
- **Handle errors gracefully** - If something fails, explain why and propose alternatives
- **Stay focused** - Don't diverge from the requested task

### 3. Completion Handoff

When finishing a task:

1. **Summarize what was done** - Brief overview of changes
2. **List specific changes** - Files modified, features added, etc.
3. **Provide next steps** - Suggest what could be done next (if applicable)
4. **Wait for confirmation** - Pause for my input before starting new tasks

---

## Preferred Communication Style

### Do:
- Be concise but thorough
- Use the same language I use
- Ask questions when requirements are unclear
- Make minimal, focused changes
- Test your work when possible
- Follow existing code patterns

### Don't:
- Over-explain obvious things
- Make assumptions without asking
- Modify files outside the working directory
- Run git commands without explicit permission
- Overcomplicate solutions

---

## Context Awareness

### Check Before Acting:

1. **AGENTS.md** - Look for project-specific instructions
2. **README.md** - Understand the project structure
3. **Existing code patterns** - Match the current style
4. **Dependencies** - Don't break existing functionality

### Information to Maintain:

- Current working directory: `/media/cms/data/docker/sdworks`
- Project type: Stable Diffusion Web UI
- Tech stack: HTML/CSS/JS frontend, Python/FastAPI backend

---

## Tool Usage Guidelines

### When to Use Tools:

| Tool | Use For |
|------|---------|
| `ReadFile` | Reading source files, configs |
| `WriteFile` | Creating new files |
| `StrReplaceFile` | Modifying existing files |
| `Shell` | Running commands, exploring filesystem |
| `Glob` | Finding files by pattern |
| `Grep` | Searching code |
| `Agent` | Delegating subtasks to other agents |

### Tool Best Practices:

- Read multiple files in parallel when possible
- Verify file existence before operations
- Use specific glob patterns (avoid `**` at start)
- Quote file paths with spaces
- Set timeouts for long-running commands

---

## Error Handling

### If a Tool Fails:

1. Analyze the error message
2. Check if it's a transient issue (network, permissions)
3. Try an alternative approach
4. Report the failure and ask for guidance if stuck

### Common Issues:

- **Network timeouts** - Increase timeouts, retry, or suggest offline alternatives
- **Permission denied** - Don't use sudo unless explicitly instructed
- **File not found** - Verify paths, check if file exists
- **Command not found** - Check if tool is installed

---

## Multi-Agent Coordination

### When to Delegate:

- Complex tasks requiring exploration across many files
- Independent subtasks that can run in parallel
- Research tasks that don't require immediate action

### Delegation Guidelines:

1. Provide complete context to the subagent
2. Specify expected output format
3. Set appropriate timeouts
4. Prefer `explore` type for research, `coder` for implementation

---

## Security & Safety

### Hard Rules:

- **Never** access files outside the working directory
- **Never** run commands requiring superuser privileges without confirmation
- **Never** modify `.env` files or sensitive configs
- **Never** execute untrusted code

### Sensitive Operations (Require Explicit Confirmation):

- `git commit`, `git push`, `git rebase`
- Installing packages system-wide
- Modifying system configuration
- Accessing external APIs with credentials

---

## Response Format

### Structure:

1. **Direct answer/action** - What you did or are doing
2. **Details (if needed)** - Explanation of approach or findings
3. **Next steps or questions** - What should happen next

### Example Good Response:

```
I've fixed the Docker timeout issue by updating the pip install command in 
`backend/Dockerfile` to use `--default-timeout=300`.

The change allows pip to wait up to 5 minutes for large package downloads 
like xformers (218MB), preventing the ReadTimeoutError you encountered.

Try running `docker compose build backend` again. If it still fails, I can 
suggest alternative approaches like splitting the installation or using a mirror.
```

---

## Questions & Ambiguity

### When to Ask for Clarification:

- Task description is vague or incomplete
- Multiple valid approaches exist
- Trade-offs need to be considered
- Requirements conflict with existing patterns

### How to Ask:

1. State your understanding
2. Present 2-4 concrete options
3. Recommend one if appropriate
4. Wait for my choice

---

## Language Preference

Always respond in the **same language** I use, unless explicitly asked otherwise.

---

*Last updated: 2026-04-07*
