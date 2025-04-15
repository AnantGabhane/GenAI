# Sure! Here is a simple Python script to perform DNS lookups using the socket library. This script will allow you to resolve a domain name to its corresponding IP address.

# Copy the code
import socket

def dns_lookup(domain):
    try:
        ip_address = socket.gethostbyname(domain)
        print(f"The IP address of {domain} is {ip_address}")
    except socket.gaierror:
        print(f"Failed to resolve {domain}")

if __name__ == "__main__":
    domain = input("Enter the domain name: ")
    dns_lookup(domain)


# This script prompts the user to enter a domain name, performs the DNS lookup, and prints the corresponding IP address. If the domain cannot be resolved, it will print an error message.