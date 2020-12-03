from typing import Optional


def regular_log(logger,
                step: int,
                tag: Optional[str] = None,
                interval: int = 100) -> None:
    tag_repr = f'[{tag}] ' if tag is not None else ''
    if step % interval == 0:
        logger.info('%sstep: %d', tag_repr, step)
