import argparse

import re

import dns.resolver

import requests

import concurrent.futures

from typing import List, Dict, Set, Optional


class SubdomainDiscovery:

    def __init__(self, base_domain: str, threads: int = 10, verbose: bool = False):
        """

        Initialize Subdomain Discovery



        :param base_domain: Main domain to discover subdomains

        :param threads: Number of concurrent threads for discovery

        :param verbose: Enable verbose output

        """

        self.base_domain = base_domain

        self.threads = threads

        self.verbose = verbose

        self.resolver = dns.resolver.Resolver()

        # Predefined lists for subdomain enumeration

        self.common_subdomains = [
            "www",
            "mail",
            "blog",
            "dev",
            "test",
            "admin",
            "support",
            "api",
            "cdn",
            "dashboard",
            "app",
            "webmail",
            "store",
            "staging",
            "beta",
            "portal",
            "login",
            "cloud",
            "remote",
            "vpn",
            "ns1",
            "ns2",
        ]

    def generate_subdomain_wordlist(self) -> List[str]:
        """

        Generate an extensive list of potential subdomains



        :return: List of potential subdomain prefixes

        """

        # Combine predefined lists with some additional variations

        additional_prefixes = [
            # Technical prefixes
            "service",
            "platform",
            "system",
            "internal",
            "external",
            "backup",
            "proxy",
            "cache",
            # Departmental prefixes
            "hr",
            "finance",
            "sales",
            "marketing",
            "engineering",
            "research",
            "support",
            # Environment prefixes
            "prod",
            "production",
            "stg",
            "staging",
            "qa",
            "uat",
            "demo",
            # Geographic prefixes
            "us",
            "eu",
            "uk",
            "ca",
            "au",
            "sg",
            "east",
            "west",
            "north",
            "south",
            # Functional prefixes
            "web",
            "online",
            "server",
            "db",
            "database",
            "mail",
            "smtp",
            "ftp",
            "ssh",
        ]

        # Combine and remove duplicates

        return list(set(self.common_subdomains + additional_prefixes))

    def resolve_subdomain(self, subdomain: str) -> Optional[Dict[str, str]]:
        """

        Resolve a specific subdomain



        :param subdomain: Subdomain to resolve

        :return: Resolved subdomain information

        """

        full_domain = f"{subdomain}.{self.base_domain}"

        try:

            # Multiple record type resolution

            record_types = ["A", "AAAA", "CNAME"]

            results = {}

            for record_type in record_types:

                try:

                    answers = self.resolver.resolve(full_domain, record_type)

                    results[record_type] = [str(rdata) for rdata in answers]

                except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):

                    continue

            if results:

                return {"domain": full_domain, "records": results}

            return None

        except Exception as e:

            if self.verbose:

                print(f"Error resolving {full_domain}: {e}")

            return None

    def discover_subdomains(self) -> List[Dict[str, str]]:
        """

        Discover subdomains using multiple techniques



        :return: List of discovered subdomains

        """

        discovered_subdomains = []

        # Generate comprehensive subdomain list

        subdomain_list = self.generate_subdomain_wordlist()

        if self.verbose:

            print(
                f"Attempting to resolve {len(subdomain_list)} potential subdomains..."
            )

        # Concurrent subdomain resolution

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:

            # Map subdomains to resolution function

            future_to_subdomain = {
                executor.submit(self.resolve_subdomain, subdomain): subdomain
                for subdomain in subdomain_list
            }

            # Collect successful resolutions

            for future in concurrent.futures.as_completed(future_to_subdomain):

                result = future.result()

                if result:

                    discovered_subdomains.append(result)

        return discovered_subdomains

    def additional_web_sources(self) -> Set[str]:
        """

        Discover subdomains from additional web sources



        :return: Set of discovered subdomains

        """

        discovered_subdomains = set()

        # List of free APIs and services for subdomain enumeration

        sources = [
            f"https://api.hackertarget.com/hostsearch/?q={self.base_domain}",
            f"https://crt.sh/?q=%.{self.base_domain}",
        ]

        for url in sources:

            try:

                if self.verbose:

                    print(f"Searching for subdomains in: {url}")

                response = requests.get(url, timeout=10)

                # Extract potential subdomains using regex

                found_subdomains = re.findall(
                    r"([a-zA-Z0-9\-]+\." + re.escape(self.base_domain) + ")",
                    response.text,
                )

                discovered_subdomains.update(found_subdomains)

            except Exception as e:

                if self.verbose:

                    print(f"Error searching {url}: {e}")

                continue

        return discovered_subdomains


def main():

    # Set up argument parser

    parser = argparse.ArgumentParser(
        description="Discover subdomains for a given domain",
        epilog="Use responsibly and respect legal and ethical guidelines.",
    )

    # Add arguments

    parser.add_argument("domain", type=str, help="Domain to discover subdomains for")

    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=10,
        help="Number of concurrent threads (default: 10)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    # Parse arguments

    args = parser.parse_args()

    # Initialize Subdomain Discovery

    subdomain_finder = SubdomainDiscovery(
        base_domain=args.domain, threads=args.threads, verbose=args.verbose
    )

    print(f"Discovering Subdomains for {args.domain}:\n")

    # DNS-based Subdomain Discovery

    print("1. DNS-based Subdomain Discovery:")

    dns_subdomains = subdomain_finder.discover_subdomains()

    for subdomain in dns_subdomains:

        print(f"Domain: {subdomain['domain']}")

        for record_type, records in subdomain["records"].items():

            print(f"  {record_type} Records: {records}")

    # Web Source Subdomain Discovery

    print("\n2. Web Source Subdomain Discovery:")

    web_subdomains = subdomain_finder.additional_web_sources()

    for domain in web_subdomains:

        print(domain)

    # Summary

    print(f"\nDiscovered {len(dns_subdomains)} DNS subdomains")

    print(f"Discovered {len(web_subdomains)} web source subdomains")


if __name__ == "__main__":

    main()
