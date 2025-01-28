from subprocess import Popen, PIPE
import bittensor as bt


def is_git_latest() -> bool:
    p = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    if err:
        return False
    current_commit = out.decode().strip()
    p = Popen(['git', 'ls-remote', 'origin', 'HEAD'], stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    if err:
        return False
    latest_commit = out.decode().split()[0]
    bt.logging.info(f'Current commit: {current_commit}, Latest commit: {latest_commit}')
    return current_commit == latest_commit
