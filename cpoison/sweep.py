import asyncio
from cpoison.main import gen_and_ft

from cpoison.protocols import UseUntrusted, HighConfTrusted, UseTrusted
from cpoison.redteams import Dumb, Honest, Lier, LongerBetter, ShorterBetter

from pathlib import Path
import json
from lowstakes.utils import gather_tqdm
import datetime


async def run():
    # protocols = [UseUntrusted(), HighConfTrusted()]
    # redteams = [
    #     Dumb(),
    #     Honest(),
    #     Lier(),
    #     LongerBetter(),
    #     ShorterBetter(),
    # ]
    protocols_and_redteams = [
        [UseTrusted(), Dumb()],
        [UseUntrusted(), Honest()],
        [UseUntrusted(), Lier()],
        [UseUntrusted(), LongerBetter()],
        [UseUntrusted(), ShorterBetter()],
        [HighConfTrusted(), Honest()],
        [HighConfTrusted(), Lier()],
        [HighConfTrusted(), LongerBetter()],
        [HighConfTrusted(), ShorterBetter()],
    ]

    save_folder = Path("data")
    save_folder.mkdir(exist_ok=True)

    async def r(protocol, redteam):
        res = await gen_and_ft(protocol, redteam)
        # time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # with ms acc
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        (save_folder / (time_str + ".json")).write_text(json.dumps(res))

    await gather_tqdm([r(p, rt) for p, rt in protocols_and_redteams], "Running protocols")


if __name__ == "__main__":
    asyncio.run(run())
