/**
 * Client-side AES-GCM vault-key utilities.
 *
 * The key is a bearer secret: anyone with this key and the ciphertext can decrypt it.
 * Never send the raw key to the BrickID backend or write it to application logs.
 */
const DB_NAME = "brickid-local-vault";
const STORE_NAME = "keys";
const KEY_ID = "vault-key-v1";

function openKeyDb(): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, 1);
        request.onupgradeneeded = () => request.result.createObjectStore(STORE_NAME);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error ?? new Error("Could not open key storage."));
    });
}

/** Generate a new extractable AES-256-GCM key so the user can export a portable backup. */
export async function createVaultKey(): Promise<CryptoKey> {
    return crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, true, ["encrypt", "decrypt"]);
}

/** Persist a CryptoKey locally; IndexedDB can store structured-cloneable CryptoKeys. */
export async function saveLocalKey(key: CryptoKey): Promise<void> {
    const db = await openKeyDb();
    await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        tx.objectStore(STORE_NAME).put(key, KEY_ID);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error ?? new Error("Could not save key."));
        tx.onabort = () => reject(tx.error ?? new Error("Key save was aborted."));
    });
    db.close();
}

/** Load the locally persisted key, or return null when no key is stored. */
export async function loadLocalKey(): Promise<CryptoKey | null> {
    const db = await openKeyDb();
    const key = await new Promise<CryptoKey | undefined>((resolve, reject) => {
        const request = db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(KEY_ID);
        request.onsuccess = () => resolve(request.result as CryptoKey | undefined);
        request.onerror = () => reject(request.error ?? new Error("Could not load key."));
    });
    db.close();
    return key ?? null;
}

/** Remove the locally stored key. This does not delete the server-side vault. */
export async function deleteLocalKey(): Promise<void> {
    const db = await openKeyDb();
    await new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, "readwrite");
        tx.objectStore(STORE_NAME).delete(KEY_ID);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error ?? new Error("Could not delete key."));
    });
    db.close();
}

/** Encrypt UTF-8 text with AES-GCM; the nonce is safe to store alongside ciphertext. */
export async function encryptText(key: CryptoKey, plaintext: string): Promise<{ nonce: string; ciphertext: string }> {
    const nonce = crypto.getRandomValues(new Uint8Array(12));
    const encrypted = await crypto.subtle.encrypt({ name: "AES-GCM", iv: nonce }, key, new TextEncoder().encode(plaintext));
    return { nonce: toBase64(nonce), ciphertext: toBase64(new Uint8Array(encrypted)) };
}

/** Decrypt UTF-8 text; authentication failure rejects rather than returning corrupted text. */
export async function decryptText(key: CryptoKey, data: { nonce: string; ciphertext: string }): Promise<string> {
    const plaintext = await crypto.subtle.decrypt(
        { name: "AES-GCM", iv: fromBase64(data.nonce) as Uint8Array<ArrayBuffer> },
        key,
        fromBase64(data.ciphertext) as Uint8Array<ArrayBuffer>,
    );
    return new TextDecoder().decode(plaintext);
}

/**
 * Export the raw key as base64 text for user-managed backup.
 * The key must be extractable to support the user-requested text backup. This weakens
 * protection against key extraction by same-origin JavaScript. The downloadable text must be treated as a master secret.
 */
export async function exportKeyText(key: CryptoKey): Promise<string> {
    const raw = await crypto.subtle.exportKey("raw", key);
    return `BRICKID-VAULT-KEY-V1\n${toBase64(new Uint8Array(raw))}\n`;
}

/** Import a key backup created by exportKeyText and return an extractable CryptoKey. */
export async function importKeyText(text: string): Promise<CryptoKey> {
    const lines = text.trim().split(/\r?\n/);
    if (lines[0] !== "BRICKID-VAULT-KEY-V1" || !lines[1]) throw new Error("This is not a valid BrickID key backup.");
    const raw = fromBase64(lines[1]) as Uint8Array<ArrayBuffer>;
    if (raw.byteLength !== 32) throw new Error("The vault key must be 256 bits.");
    return crypto.subtle.importKey("raw", raw, { name: "AES-GCM" }, true, ["encrypt", "decrypt"]);
}

function toBase64(bytes: Uint8Array): string {
    let binary = "";
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return btoa(binary);
}

function fromBase64(value: string): Uint8Array {
    const binary = atob(value);
    return Uint8Array.from(binary, (char) => char.charCodeAt(0));
}
