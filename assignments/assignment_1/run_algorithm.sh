# run this script in the terminal to generate the results for assignment 1

for seed in 1 2 3 4 5; do
    uv run assignments/assignment_1/group24_algorithm.py --pop 50 --budget 100 --mutation-weights 0.7 0.1 0.1 0.1 --seed $seed --run-name variantA --no-visualize
    uv run assignments/assignment_1/group24_algorithm.py --pop 50 --budget 100 --mutation-weights 0.1 0.7 0.1 0.1 --seed $seed --run-name variantB --no-visualize
    uv run assignments/assignment_1/group24_algorithm.py --pop 50 --budget 100 --baseline --seed $seed --run-name baseline
done