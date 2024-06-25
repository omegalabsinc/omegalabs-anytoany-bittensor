import asyncio
from leaderboard import pull_and_cache_miner_info, pull_and_cache_recent_descriptions

async def periodic_task(interval, *tasks):
    while True:
        try:
            await asyncio.gather(*[task() for task in tasks])
        except Exception as err:
            print("Error during syncing data", str(err))
        await asyncio.sleep(interval)

async def main():
    await periodic_task(1800, pull_and_cache_miner_info, pull_and_cache_recent_descriptions)

if __name__ == "__main__":
    asyncio.run(main())