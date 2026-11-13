"""
Shared constants for the IDTAP library.
"""

# Fundamental frequency validation constants
MIN_FUNDAMENTAL_HZ = 50   # Minimum reasonable fundamental frequency for musical analysis
MAX_FUNDAMENTAL_HZ = 400  # Maximum reasonable fundamental frequency for musical analysis
# idtap-contract version this client implements (idtap-contract contract.json
# `version`). It is NOT the package version: idtap 0.1.59 implements 0.3.0.
CONTRACT_VERSION = "0.3.0"

# PROP-8 write guard: every request carries `X-IDTAP-Client: python/<contract
# version>`. The server refuses a save from a client below 0.3.0 when the
# stored piece is not legacy-safe, and uses the header as its CSRF check.
CLIENT_HEADER = "X-IDTAP-Client"
CLIENT_HEADER_VALUE = f"python/{CONTRACT_VERSION}"
