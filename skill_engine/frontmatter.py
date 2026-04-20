from __future__ import annotations


class SkillFrontmatterError(ValueError):
    pass


def split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        raise SkillFrontmatterError("SKILL.md must start with YAML frontmatter delimited by ---")
    closing = text.find("\n---\n", 4)
    if closing == -1:
        raise SkillFrontmatterError("Could not find closing YAML frontmatter delimiter")
    return text[4:closing], text[closing + 5 :]
