from pyspark import SparkContext

print("Starting Spark PageRank...")

sc = SparkContext("local[*]", "PageRank")

# Path to dataset (update if needed)
file_path = "file:///home/student/ml_assignment4/PySpark-PageRank/graphs/whole.txt"

# Load edges
lines = sc.textFile(file_path)

# Parse edges
edges = lines.map(lambda line: line.split()) \
             .map(lambda parts: (int(parts[0]), int(parts[1])))

# Remove duplicate edges
edges = edges.distinct()

# Create adjacency list
links = edges.groupByKey()

# Initialize ranks
nodes = links.map(lambda x: x[0])
ranks = nodes.map(lambda node: (node, 1.0))

print("Graph loaded successfully!")

print("\nRunning PageRank iterations...")

beta = 0.8
num_iterations = 40

# Prepare links (adjacency list)
links = edges.groupByKey().mapValues(list)

# Initialize ranks
nodes = links.map(lambda x: x[0])
n = nodes.count()

ranks = nodes.map(lambda node: (node, 1.0 / n))

for i in range(num_iterations):
    contributions = links.join(ranks).flatMap(
        lambda x: [(dest, x[1][1] / len(x[1][0])) for dest in x[1][0]]
    )

    ranks = contributions.reduceByKey(lambda x, y: x + y) \
        .mapValues(lambda rank: (1 - beta) / n + beta * rank)

print("PageRank iterations completed!")

print("\nExtracting Top 5 and Bottom 5 nodes...")

# Collect results
rank_list = ranks.collect()

# Sort descending
rank_list_sorted = sorted(rank_list, key=lambda x: x[1], reverse=True)

print("\nTop 5 nodes:")
for node, score in rank_list_sorted[:5]:
    print(f"{node}: {score:.6f}")

print("\nBottom 5 nodes:")
for node, score in rank_list_sorted[-5:]:
    print(f"{node}: {score:.6f}")

print("\nStep Q3 completed ✅")