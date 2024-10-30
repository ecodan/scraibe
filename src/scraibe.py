import argparse
from pathlib import Path

from src.conductor import Conductor, PaperbackWriter, HistoryPodcaster
from src.logutils import create_logger

logger = create_logger("scrAIbe")

VALID_OPERATIONS: list[str] = ["develop", "draft"]

if __name__ == '__main__':
    """
    Main entry point via CLI
    
    Invoked by (e.g.) 
    `python scraibe.py longform-fiction /path/to/working_dir -e bedrock -o develop`
    
    """
    logger.info("Starting scraibe")
    parser = argparse.ArgumentParser(description='scrAIbe launcher')
    parser.add_argument('generate', choices=['longform-fiction', 'podcast'],
                        help='What to generate [longform-fiction|podcast]')
    parser.add_argument('working_dir', type=str,
                        help='Path to parent location of working directories')
    parser.add_argument('-e', '--env', type=str, default='local',
                        help='LLM environment to use [local|bedrock]')
    parser.add_argument('-o', '--operations', nargs='+', default=['develop'],
                        help=f'Generation steps to execute (default: develop). Valid options: {VALID_OPERATIONS}')
    parser.add_argument('-p', '--project_name', type=str, default=None,
                        help='Name of the project working directory (only needed if drafting without first generating)')

    args = parser.parse_args()

    # ensure working dir is valid
    working_dir: Path = Path(args.working_dir)
    assert working_dir.is_dir()
    logger.info(f"Working dir={working_dir}")

    # instantiate conductor
    conductor: Conductor | None = None
    if args.generate == 'longform-fiction':
        conductor = PaperbackWriter(working_dir=working_dir, env=args.env)
    elif args.generate == 'podcast':
        conductor = HistoryPodcaster(working_dir=working_dir, env=args.env)
    else:
        raise ValueError('no valid generation option provided')

    logger.info(f"Generating {args.generate}")

    # ensure all operations are valid
    for operation in args.operations:
        if operation not in VALID_OPERATIONS:
            raise ValueError(f"invalid generation option: {operation}")

    # execute operations
    project_dir: Path | None = None
    if 'develop' in args.operations:
        logger.info(f"Developing concept...")
        project_dir = conductor.develop_concept()
    if 'draft' in args.operations:
        logger.info(f"Creating draft...")
        if not project_dir:
            project_dir = working_dir / args.project_name
            assert project_dir.is_dir(), f"{project_dir} does not exist"
        conductor.draft_narrative(project_dir)

    logger.info("Done")
