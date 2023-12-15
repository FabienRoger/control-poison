import asyncio
from cpoison.main import gen_and_ft

from cpoison.protocols import UseUntrusted
from cpoison.redteams import Dumb, Honest, Lier, LongerBetter, ShorterBetter

from pathlib import Path
import json
from lowstakes.utils import gather_tqdm


async def run():
    protocol = UseUntrusted()
    redteams = [
        Dumb(),
        Honest(),
        Lier(),
        LongerBetter(),
        ShorterBetter(),
    ]

    save_folder = Path("data")
    save_folder.mkdir(exist_ok=True)

    async def r(redteam):
        res = await gen_and_ft(protocol, redteam)
        (save_folder / (res["short_id"] + ".json")).write_text(json.dumps(res))

    await gather_tqdm([r(rt) for rt in redteams], "Generating data")


if __name__ == "__main__":
    asyncio.run(run())
