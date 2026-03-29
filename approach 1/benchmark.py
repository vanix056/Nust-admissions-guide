from statistics import mean
from time import perf_counter

from main import build_default_chatbot


def run_benchmark(iterations: int = 20) -> None:
    bot = build_default_chatbot()
    queries = [
        "is there age limit for undergraduate admission",
        "can i apply for more than one programme",
        "what is fee structure for mbbs programme",
        "is there hostel facility and what is fee structure for mbbs",
        "do you offer aeronautical pilot license",
    ]

    times = []
    for _ in range(iterations):
        for query in queries:
            start = perf_counter()
            _ = bot.answer(query)
            times.append((perf_counter() - start) * 1000)

    print(f"runs: {len(times)}")
    print(f"avg_ms: {mean(times):.2f}")
    print(f"max_ms: {max(times):.2f}")


if __name__ == "__main__":
    run_benchmark()

