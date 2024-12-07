import google.auth
from google.auth.transport.requests import Request
from google.cloud import compute_v1

def fetch_ip_addresses(project_id):
    """Fetch both static and ephemeral IP addresses from a Google Cloud project."""
    credentials, _ = google.auth.default()
    credentials.refresh(Request())

    address_client = compute_v1.AddressesClient()
    instance_client = compute_v1.InstancesClient()
    region_client = compute_v1.RegionsClient()
    zone_client = compute_v1.ZonesClient()
    
    all_addresses = []

    # Fetch regional static IP addresses
    region_list = region_client.list(project=project_id)
    for region in region_list:
        region_name = region.name.split('/')[-1]
        try:
            addresses = address_client.list(project=project_id, region=region_name)
            for address in addresses:
                all_addresses.append({
                    'name': address.name,
                    'address': address.address,
                    'region': region_name,
                    'type': 'static',
                    'status': address.status
                })
        except Exception as e:
            print(f"Error fetching static addresses for region {region_name}: {str(e)}")

    # Fetch global static IP addresses
    try:
        global_addresses = address_client.list(project=project_id)
        for address in global_addresses:
            all_addresses.append({
                'name': address.name,
                'address': address.address,
                'region': 'global',
                'type': 'static',
                'status': address.status
            })
    except Exception as e:
        print(f"Error fetching global static addresses: {str(e)}")

    # Fetch ephemeral IP addresses (associated with VM instances)
    zone_list = zone_client.list(project=project_id)
    for zone in zone_list:
        zone_name = zone.name.split('/')[-1]
        try:
            instances = instance_client.list(project=project_id, zone=zone_name)
            for instance in instances:
                for network_interface in instance.network_interfaces:
                    if network_interface.access_configs:
                        for access_config in network_interface.access_configs:
                            if access_config.nat_ip:
                                all_addresses.append({
                                    'name': f"{instance.name}-{network_interface.name}",
                                    'address': access_config.nat_ip,
                                    'zone': zone_name,
                                    'type': 'ephemeral',
                                    'status': instance.status
                                })
        except Exception as e:
            print(f"Error fetching ephemeral addresses for zone {zone_name}: {str(e)}")

    return all_addresses

# Usage
project_id = 'kr-10345-pznpfm-d'
ip_addresses = fetch_ip_addresses(project_id)

# Save to file
import json
with open('gcp_ip_addresses.json', 'w') as f:
    json.dump(ip_addresses, f, indent=2)

print(f"Saved {len(ip_addresses)} IP addresses to gcp_ip_addresses.json")