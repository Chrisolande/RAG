"""
Add this at the beginning of your notebook or script to force IPv4
"""
import socket
# Force IPv4
original_getaddrinfo = socket.getaddrinfo

def getaddrinfo_ipv4_only(*args, **kwargs):
    responses = original_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]

socket.getaddrinfo = getaddrinfo_ipv4_only

print("IPv6 has been disabled for this application. It will only use IPv4 connections.")
