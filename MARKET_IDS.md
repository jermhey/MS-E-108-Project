# Kalshi & Polymarket — Spotify Top Artist Market Identifiers

**Event:** Top Spotify artist by monthly listeners (winner-take-all)  
**Resolution:** Artist with greatest monthly listeners on Spotify at month-end.

---

## Polymarket (February 2026)

**Event slug:** `top-spotify-artist-this-month`  
**Event ID:** `198671`  
**URL:** https://polymarket.com/event/top-spotify-artist-this-month

### Outcome markets (condition IDs)

| Artist        | conditionId |
|---------------|-------------|
| Bruno Mars    | `0xd4d04dbaf238a3aa7fe9333caafdf45d6fcc91d5de4e7cd0257d130e89444e68` |
| Noah Kahan    | `0xe78fda3186bd5beadb6a4d052ab303ea2a22a348cf2daca1c12a9b2a67f0a23f` |
| Bad Bunny     | `0x699a92c550395c9375043a319b7e7e9ce96ad80f39ac0ec916daaf5b49ccb684` |
| The Weeknd    | `0xe8ec9fb94e2c3c30a53bd6cddd5ed0818430576c109028ec04ea6fe47ec16ffd` |
| Kendrick Lamar| `0xaeb2dffde90375ad17f6c7248d27325553eccd84d4dcddaef23bc4813ab2c3c8` |
| Taylor Swift  | `0xe15530ae415054237e1cb5b970020579408b028196fe8a315c4f301d96e2e44e` |
| Drake         | `0x58774a6d5900caf17bbc49bf569d3652098e1550db5c3e997b2673c568dad666` |
| Billie Eilish | `0x8260bbc55b4c6efd6917cd788a964587d318d59e3442f4f0632f6ed779ce58b4` |
| Lady Gaga     | `0x894cbeb65515fe07bbaf23cd9ba711ad4b5dd81b82f570bc7f344daae47186b9` |
| Kanye West    | `0x8079bf45edbb00b4cbec1f03c15cb040222735f3d436bf9c78dbfcbd90cde7c1` |
| Rihanna       | `0xd7e06e467c7c0582c7d79cc5e8ad33a63a45d365c86705902d2f424848e2bbeb` |
| Ed Sheeran    | `0x9f32fd8f151686a80e37cd0260669d206e04209e55cba34fca33aaa08c2c20b3` |
| Coldplay      | `0xb6eb8958db15dcfbd39920e3c71d9f99ff4d3e0c0c0956358aed3b456c9fa1f3` |

**API:** `GET https://gamma-api.polymarket.com/events?slug=top-spotify-artist-this-month`

---

## Kalshi (KXTOPMONTHLY)

**Format:** `KXTOPMONTHLY-{YY}{MON}-{ARTIST_SUFFIX}`

### Artist suffix mapping

| Suffix | Artist        | Polymarket equivalent |
|--------|---------------|------------------------|
| BAD    | Bad Bunny     | Bad Bunny             |
| BRU    | Bruno Mars    | Bruno Mars            |
| WEE    | The Weeknd    | The Weeknd            |
| TAY    | Taylor Swift  | Taylor Swift          |
| ARI    | Ariana Grande | *(no Feb 2026 outcome)* |
| BIL    | Billie Eilish | Billie Eilish         |
| DRA    | Drake         | Drake                 |
| KEN    | Kendrick Lamar| Kendrick Lamar        |
| LAD    | Lady Gaga     | Lady Gaga             |
| RIH    | Rihanna       | Rihanna               |
| ED     | Ed Sheeran    | Ed Sheeran            |
| COL    | Coldplay      | Coldplay              |
| COLD   | Coldplay      | Coldplay              |

### February 2026 (26FEB) — direct Polymarket overlap

| Kalshi ticker         | Polymarket conditionId |
|-----------------------|------------------------|
| KXTOPMONTHLY-26FEB-BAD| `0x699a92c550395c9375043a319b7e7e9ce96ad80f39ac0ec916daaf5b49ccb684` |
| KXTOPMONTHLY-26FEB-BRU| `0xd4d04dbaf238a3aa7fe9333caafdf45d6fcc91d5de4e7cd0257d130e89444e68` |
| KXTOPMONTHLY-26FEB-WEE| `0xe8ec9fb94e2c3c30a53bd6cddd5ed0818430576c109028ec04ea6fe47ec16ffd` |
| KXTOPMONTHLY-26FEB-TAY| `0xe15530ae415054237e1cb5b970020579408b028196fe8a315c4f301d96e2e44e` |
| KXTOPMONTHLY-26FEB-BIL| `0x8260bbc55b4c6efd6917cd788a964587d318d59e3442f4f0632f6ed779ce58b4` |
| KXTOPMONTHLY-26FEB-DRA| `0x58774a6d5900caf17bbc49bf569d3652098e1550db5c3e997b2673c568dad666` |
| KXTOPMONTHLY-26FEB-KEN| `0xaeb2dffde90375ad17f6c7248d27325553eccd84d4dcddaef23bc4813ab2c3c8` |
| KXTOPMONTHLY-26FEB-LAD| `0x894cbeb65515fe07bbaf23cd9ba711ad4b5dd81b82f570bc7f344daae47186b9` |
| KXTOPMONTHLY-26FEB-RIH| `0xd7e06e467c7c0582c7d79cc5e8ad33a63a45d365c86705902d2f424848e2bbeb` |
| KXTOPMONTHLY-26FEB-ED | `0x9f32fd8f151686a80e37cd0260669d206e04209e55cba34fca33aaa08c2c20b3` |
| KXTOPMONTHLY-26FEB-COL| `0xb6eb8958db15dcfbd39920e3c71d9f99ff4d3e0c0c0956358aed3b456c9fa1f3` |

### All cached Kalshi tickers (86 contracts)

```
KXTOPMONTHLY-25AUG-BAD   KXTOPMONTHLY-25AUG-BRU   KXTOPMONTHLY-25AUG-ED
KXTOPMONTHLY-25AUG-LAD   KXTOPMONTHLY-25AUG-TAY   KXTOPMONTHLY-25AUG-WEE
KXTOPMONTHLY-25DEC-ARI   KXTOPMONTHLY-25DEC-BAD   KXTOPMONTHLY-25DEC-BIL
KXTOPMONTHLY-25DEC-BRU   KXTOPMONTHLY-25DEC-COL   KXTOPMONTHLY-25DEC-DRA
KXTOPMONTHLY-25DEC-ED    KXTOPMONTHLY-25DEC-JUS   KXTOPMONTHLY-25DEC-KEN
KXTOPMONTHLY-25DEC-LAD   KXTOPMONTHLY-25DEC-MAR   KXTOPMONTHLY-25DEC-MIC
KXTOPMONTHLY-25DEC-RIH   KXTOPMONTHLY-25DEC-TAY   KXTOPMONTHLY-25DEC-WEE
KXTOPMONTHLY-25JUL-BIL   KXTOPMONTHLY-25JUL-BRU   KXTOPMONTHLY-25JUL-COLD
KXTOPMONTHLY-25JUL-DRA   KXTOPMONTHLY-25JUL-TAY   KXTOPMONTHLY-25JUL-WEE
KXTOPMONTHLY-25JUN-BBUN  KXTOPMONTHLY-25JUN-BEIL  KXTOPMONTHLY-25JUN-BMAR
KXTOPMONTHLY-25JUN-COLD  KXTOPMONTHLY-25JUN-ESHE  KXTOPMONTHLY-25JUN-KLAM
KXTOPMONTHLY-25JUN-LGAG  KXTOPMONTHLY-25JUN-RIHA  KXTOPMONTHLY-25JUN-TSWI
KXTOPMONTHLY-25JUN-TWEE  KXTOPMONTHLY-25NOV-BAD   KXTOPMONTHLY-25NOV-BIL
KXTOPMONTHLY-25NOV-BRU   KXTOPMONTHLY-25NOV-COL   KXTOPMONTHLY-25NOV-DRA
KXTOPMONTHLY-25NOV-ED    KXTOPMONTHLY-25NOV-KEN   KXTOPMONTHLY-25NOV-LAD
KXTOPMONTHLY-25NOV-RIH   KXTOPMONTHLY-25NOV-TAY   KXTOPMONTHLY-25NOV-WEE
KXTOPMONTHLY-25OCT-BAD   KXTOPMONTHLY-25OCT-BIL   KXTOPMONTHLY-25OCT-BRU
KXTOPMONTHLY-25OCT-TAY   KXTOPMONTHLY-25OCT-WEE   KXTOPMONTHLY-25SEP-BAD
KXTOPMONTHLY-25SEP-BRU   KXTOPMONTHLY-25SEP-ED    KXTOPMONTHLY-25SEP-TAY
KXTOPMONTHLY-25SEP-WEE   KXTOPMONTHLY-26AUG-BAD   KXTOPMONTHLY-26AUG-BIL
KXTOPMONTHLY-26AUG-BRU   KXTOPMONTHLY-26AUG-DRA   KXTOPMONTHLY-26AUG-WEE
KXTOPMONTHLY-26FEB-BAD   KXTOPMONTHLY-26FEB-BIL   KXTOPMONTHLY-26FEB-BRU
KXTOPMONTHLY-26FEB-COL   KXTOPMONTHLY-26FEB-DRA   KXTOPMONTHLY-26FEB-ED
KXTOPMONTHLY-26FEB-KEN   KXTOPMONTHLY-26FEB-LAD   KXTOPMONTHLY-26FEB-RIH
KXTOPMONTHLY-26FEB-TAY   KXTOPMONTHLY-26FEB-WEE   KXTOPMONTHLY-26JAN-ARI
KXTOPMONTHLY-26JAN-BAD   KXTOPMONTHLY-26JAN-BIL   KXTOPMONTHLY-26JAN-BRU
KXTOPMONTHLY-26JAN-COL   KXTOPMONTHLY-26JAN-DRA   KXTOPMONTHLY-26JAN-ED
KXTOPMONTHLY-26JAN-KEN   KXTOPMONTHLY-26JAN-LAD   KXTOPMONTHLY-26JAN-RIH
KXTOPMONTHLY-26JAN-TAY   KXTOPMONTHLY-26JAN-WEE
```

---

## Kalshi — LA Highest Temperature (KXHIGHLAX)

**Format:** `KXHIGHLAX-{YY}{MON}{DD}-{BRACKET}`  
**Resolution:** NWS Climatological Report (Daily) for LAX Airport (33.93806N, 118.38889W)  
**URL:** https://kalshi.com/markets/kxhighlax/highest-temperature-in-los-angeles

### Bracket format

| Suffix | Meaning | Example |
|--------|---------|---------|
| T{N}   | High temp **below** N°F | KXHIGHLAX-26MAR01-T71 → "Will high be <71°?" |
| B{N}   | High temp in **bracket** around N°F | KXHIGHLAX-26MAR01-B73.5 → "Will high be 73-74°?" |
| T{N} (high) | High temp **above** N°F | KXHIGHLAX-26MAR01-T78 → "Will high be >78°?" |

### Dataset stats

- **Daily events:** ~6 bracket contracts per day
- **Settled events:** 419+ (running since ~Jan 2025)
- **Data pulled:** 60 most recent settled events, **505,324 trades** across 360 markets
- **Average trades/contract:** ~1,400 (10× more liquid than KXTOPMONTHLY)

### Polymarket equivalent

**No Polymarket equivalent exists.** Polymarket runs temperature markets for NYC (LaGuardia) and Dallas, but does not currently offer an LA/LAX temperature market.

---

*Generated from Polymarket gamma-api and Kalshi cache. Polymarket condition IDs may change if the event is recreated.*
