<!--
# Todo List File Format Tutorial

This file is a Todo List file that follows specific formatting rules:

## Task Format
- Tasks are written using markdown checkboxes: "- [ ]" for incomplete tasks, "- [x]" for completed tasks
- Each task can have:
  - A priority: @priority(high|medium|low)
  - A due date: @date(YYYY-MM-DD)
  - A description: any indented text below the task
  - Sub-tasks: indented tasks below the parent task

## Example:
- [ ] Main task with high priority @priority(high) @date(2024-01-01)
  Description of the main task
  Can be multiple lines
  - [ ] Sub-task 1
  - [x] Completed sub-task 2

## Special Sections
- "## Linked Todo Files" section is reserved for linking other todo files
- HTML comments <!-- --> are ignored by the parser

## Tips
- Use headings (# and ##) to organize your tasks
- Keep the formatting consistent for better readability
- IDs are automatically added to tasks when saved
-->

# Todo List

<!-- Format de la Todo List :
- [ ] Tâche @id(abc123)
  - [ ] Sous-tâche @id(def456)
-->

