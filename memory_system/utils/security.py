"""Security utilities for encryption and signing operations."""
from typing import Tuple, Union

from cryptography.fernet import Fernet
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

def generate_symmetric_key() -> bytes:
    """Generate a new Fernet (symmetric) encryption key."""
    return Fernet.generate_key()

def encrypt_data(data: Union[str, bytes], key: Union[str, bytes]) -> bytes:
    """
    Encrypt data using a Fernet symmetric key.
    
    Args:
        data: The plaintext data to encrypt (string or bytes).
        key: The Fernet key to use for encryption (string or bytes).
    
    Returns:
        The encrypted data (bytes).
    """
    key_bytes = key.encode() if isinstance(key, str) else key
    data_bytes = data.encode() if isinstance(data, str) else data
    fernet = Fernet(key_bytes)
    return fernet.encrypt(data_bytes)

def decrypt_data(token: bytes, key: Union[str, bytes]) -> bytes:
    """
    Decrypt data using a Fernet symmetric key.
    
    Args:
        token: The encrypted data (bytes).
        key: The Fernet key used for encryption (string or bytes).
    
    Returns:
        The decrypted data (bytes).
    """
    key_bytes = key.encode() if isinstance(key, str) else key
    fernet = Fernet(key_bytes)
    return fernet.decrypt(token)

def generate_rsa_key_pair(key_size: int = 2048) -> Tuple[bytes, bytes]:
    """
    Generate an RSA private/public key pair.
    
    Args:
        key_size: The length of the RSA key in bits (default: 2048).
    
    Returns:
        A tuple (private_key_bytes, public_key_bytes) in PEM format.
    """
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private_pem, public_pem

def sign_data(data: Union[str, bytes], private_key_pem: Union[str, bytes]) -> bytes:
    """
    Sign data using an RSA private key with PSS padding (SHA-256).
    
    Args:
        data: The data to sign (string or bytes).
        private_key_pem: The RSA private key in PEM format (string or bytes).
    
    Returns:
        The signature (bytes).
    """
    data_bytes = data.encode() if isinstance(data, str) else data
    key_bytes = private_key_pem.encode() if isinstance(private_key_pem, str) else private_key_pem
    private_key = serialization.load_pem_private_key(key_bytes, password=None)
    signature = private_key.sign(
        data_bytes,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return signature

def verify_signature(data: Union[str, bytes], signature: bytes, public_key_pem: Union[str, bytes]) -> bool:
    """
    Verify an RSA signature using PSS padding (SHA-256).
    
    Args:
        data: The original data that was signed (string or bytes).
        signature: The signature to verify (bytes).
        public_key_pem: The RSA public key in PEM format (string or bytes).
    
    Returns:
        True if the signature is valid, False otherwise.
    """
    data_bytes = data.encode() if isinstance(data, str) else data
    key_bytes = public_key_pem.encode() if isinstance(public_key_pem, str) else public_key_pem
    public_key = serialization.load_pem_public_key(key_bytes)
    try:
        public_key.verify(
            signature,
            data_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except InvalidSignature:
        return False
    except Exception:
        return False
