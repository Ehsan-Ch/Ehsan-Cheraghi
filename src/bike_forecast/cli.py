"""Command line interface; models and reports are written only by explicit commands."""
import argparse
import json
from pathlib import Path
from .data import download
from .experiment import run
from .reporting import report


def main(argv=None):
    parser = argparse.ArgumentParser(description='Resumable bike demand forecasting')
    sub = parser.add_subparsers(dest='command', required=True)
    get = sub.add_parser('download', help='Download and verify the pinned public UCI archive')
    get.add_argument('--data', type=Path, default=Path('data/raw/bike-sharing.zip'))
    for name in ['run', 'verify']:
        command = sub.add_parser(name)
        command.add_argument('--data', type=Path, default=Path('data/raw/bike-sharing.zip'))
        command.add_argument('--output', type=Path, default=Path('artifacts/official'))
        if name == 'run':
            command.add_argument('--max-stages', type=int, default=None)
    status = sub.add_parser('status')
    status.add_argument('--output', type=Path, default=Path('artifacts/official'))
    export = sub.add_parser('report')
    export.add_argument('--output', type=Path, default=Path('artifacts/official'))
    export.add_argument('--destination', type=Path, default=Path('reports'))
    args = parser.parse_args(argv)
    if args.command == 'download':
        print(download(args.data))
    elif args.command == 'status':
        path = args.output / 'state.json'
        print(path.read_text() if path.exists() else json.dumps({'status': 'not_completed_or_paused_yet'}))
    elif args.command == 'report':
        report(args.output, args.destination)
        print(args.destination / 'RESULTS.md')
    else:
        maximum = 0 if args.command == 'verify' else args.max_stages
        if maximum is not None and maximum < 0:
            parser.error('--max-stages must be nonnegative')
        _, state = run(args.data, args.output, max_stages=maximum)
        print(json.dumps(state, indent=2))
        if args.command == 'verify' and state['status'] != 'complete':
            raise SystemExit('Verification incomplete: some fitting stages are missing')


if __name__ == '__main__':
    main()
