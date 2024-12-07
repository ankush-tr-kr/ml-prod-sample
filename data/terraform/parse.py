import re
import pandas as pd

def parse_terraform_log(log_content):
    # Initialize data structures to store parsed information
    resources_to_add = []
    resources_to_change = []
    resources_to_destroy = []

    # Split the log content into lines
    lines = log_content.split("\n")

    # Iterate over each line
    for line in lines:
        # Check if the line indicates a resource action
        if line.startswith("  + resource"):
            resource_info = extract_resource_info(lines, line)
            if resource_info:
                resources_to_add.append(resource_info)
        elif line.startswith("  ~ resource"):
            resource_info = extract_resource_info(lines, line)
            if resource_info:
                resources_to_change.append(resource_info)
        elif line.startswith("  - resource"):
            resource_info = extract_resource_info(lines, line)
            if resource_info:
                resources_to_destroy.append(resource_info)

    # Return the parsed information
    return {
        "resources_to_add": resources_to_add,
        "resources_to_change": resources_to_change,
        "resources_to_destroy": resources_to_destroy
    }

def extract_resource_info(lines, start_line):
    resource_info = {}
    resource_type = re.search(r'"(.+?)"', start_line).group(1)
    resource_info["resource_type"] = resource_type

    # Iterate over the lines following the resource declaration
    for line in lines[lines.index(start_line) + 1:]:
        if not line.startswith("      "):
            break

        # Extract cluster_id, instance_id, or name if available
        if "cluster_id" in line:
            cluster_id = re.search(r'= "(.+?)"', line)
            if cluster_id and "(known after apply)" not in cluster_id.group(1):
                resource_info["cluster_id"] = cluster_id.group(1)
        elif "instance_id" in line:
            instance_id = re.search(r'= "(.+?)"', line)
            if instance_id and "(known after apply)" not in instance_id.group(1):
                resource_info["instance_id"] = instance_id.group(1)
        elif "name" in line:
            name = re.search(r'= "(.+?)"', line)
            if name and "(known after apply)" not in name.group(1):
                resource_info["name"] = name.group(1)

    return resource_info

# Example usage
log_file = "terraform-120624.log"

with open(log_file, "r") as file:
    log_content = file.read()

parsed_data = parse_terraform_log(log_content)

# print("Resources to add:")
# for resource in parsed_data["resources_to_add"]:
#     print(resource)

# print("\nResources to change:")
# for resource in parsed_data["resources_to_change"]:
#     print(resource)

# print("\nResources to destroy:")
# for resource in parsed_data["resources_to_destroy"]:
#     print(resource)

# Create a pandas DataFrame for each resource category
df_add = pd.DataFrame(parsed_data["resources_to_add"])
df_change = pd.DataFrame(parsed_data["resources_to_change"])
df_destroy = pd.DataFrame(parsed_data["resources_to_destroy"])

# Concatenate the DataFrames vertically
df_combined = pd.concat([df_add, df_change, df_destroy], keys=["Resources to add", "Resources to change", "Resources to destroy"])

# Reset the index to flatten the DataFrame
df_combined = df_combined.reset_index(level=1, drop=True).reset_index()

# Rename the 'index' column to 'Category'
df_combined = df_combined.rename(columns={"index": "Category"})

# Export the combined DataFrame to a CSV file
df_combined.to_csv("terraform_resources.csv", index=False)

print("Exported data to terraform_resources.csv")