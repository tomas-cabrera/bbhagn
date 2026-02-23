# Array

Folder for running jobs as sbatch array.

## Run descriptions

| Run | GWs | Mass cut | AGN (astrophysical) | AGN (observed) | Flare rate | 
| --- | --- | -------- | ------------------- | -------------- | ---------- |
| 1   | All | $0 < m_1/m_\odot < 200$ | $10^{-4.75}$ | * | $1.06 \times 10^{-8}$ |
| 2   | ^ | ^ | ^ | * | Kimura+20 |
| 3   | ^ | ^ | QLFHopkins, $m_g < 20.5$ | * | $1.06 \times 10^{-8}$ |
| 4   | ^ | ^ | ^ | * | Kimura+20 |
| 5   | ^ | ^ | $10^{-4.75}$ | * | $4.79 \times 10^{-8}$ |
| 7   | ^ | ^ | QLFHopkins, $m_g < 20.5$ | * | ^ |
| 9   | ^ | ^ | Milliquas | * | $1.06 \times 10^{-8}$ |
| 10  | ^ | ^ | ^ | * | $4.79 \times 10^{-8}$ |
| 11  | ^ | ^ | ^ | QLFHopkins, $m_g < 20.5$ | $1.06 \times 10^{-8}$ |
| 12  | ^ | $0 < m_1/m_\odot < 40$ | ^ | ^ | ^ |
| 13  | ^ | $40 < m_1/m_\odot < 200$ | ^ | ^ | ^ |
| 14  | ^ | $0 < m_{\rm fin}/m_\odot < 40$ | ^ | ^ | ^ |
| 15  | ^ | $40 < m_{\rm fin}/m_\odot < 200$ | ^ | ^ | ^ |
| 16  | ^ | $0 < m_1/m_\odot < 200$ | ^ | QLFHopkins, $L_{\rm bol}/L_\odot > 3 \times 10^{42}$ | ^ |
| 17  | ^ | ^ | ^ | QLFHopkins, $L_{\rm bol}/L_\odot > 5 \times 10^{41}$ | ^ |
| 18  | Flare-coincident | ^ | ^ | QLFHopkins, $m_g < 20.5$ | ^ |
| 19  | Flare-coincident | ^ | ^ | QLFHopkins, $L_{\rm bol}/L_\odot > 3 \times 10^{42}$ | ^ |
| 20  | Flare-coincident | ^ | ^ | QLFHopkins, $L_{\rm bol}/L_\odot > 5 \times 10^{41}$ | ^ |

*: The observed distribution is the same as the astrophysical

^: The entry is the same as the previous line

11_planck is 11 done with Planck cosmology
11_shoes is 11 donew with SH0ES cosmology
