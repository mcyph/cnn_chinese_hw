# Third-Party Data & Attribution

This recognizer is trained and validated on several third-party stroke corpora.
Each is credited below as required by its license. License texts live alongside
the data under `cnn_chinese_hw/data/`.

## License-separated models

Because CC BY-SA 3.0, LGPL 2.1 and the Arphic PL are mutually copyleft-
incompatible, the corpora are **not co-mingled in one set of weights**. Each
*license group* is trained into its own checkpoint, and the models are combined
only at inference time (summed calibrated probabilities — see `recognizer.py`).
A model's weights derive solely from its *training* corpora; its *validation*
corpus is held out (early-stopping / calibration only) and never enters the
weights.

| Model (checkpoint) | Trains on | Validated on (held out) | Weights license |
|---|---|---|---|
| `hw_model.permissive.pt` | author + Tomoe (LGPL 2.1) + Make Me a Hanzi (Arphic PL) | KanjiVG | LGPL 2.1 / Arphic PL — **no CC BY-SA** |
| `hw_model.ccbysa.pt` | author + KanjiVG (CC BY-SA 3.0) | Tomoe | **CC BY-SA 3.0** (share-alike) |

So the **permissive model is free of any share-alike obligation** and can be
shipped commercially under the project's LGPL/APL terms. The **ccbysa model**, if
distributed, carries CC BY-SA 3.0 (attribution + share-alike). Ship whichever
models your licensing allows; the recognizer ensembles whatever is present.

| Corpus | License | Commercial use |
|---|---|---|
| Tomoe | LGPL 2.1 | ✅ (copyleft) |
| Make Me a Hanzi (`graphics.txt`) | Arphic Public License | ✅ |
| KanjiVG | CC BY-SA 3.0 | ✅ (**share-alike**) |

> Deliberately **excluded** for licensing reasons: CASIA-OLHWDB / CASIA-HWDB and
> the ETL Character Database (research / non-commercial only), Tegaki / Zinnia
> models and the OpenHandWrite toolbox (GPL), and the AI-FREE-Team Traditional
> Chinese Handwriting Dataset (CC BY-**NC**-SA 4.0, non-commercial).

---

## Tomoe

Online handwriting strokes for Chinese (zh / zh-TW) and Japanese, used as the
primary training corpus.

- Source: <https://sourceforge.net/projects/tomoe/>
- License: GNU Lesser General Public License, version 2.1.

## Make Me a Hanzi — `graphics.txt`

Per-character stroke **median** poly-lines (synthetic pen trajectories) for
~9,500 characters, used as an additional training source to broaden character
coverage. `graphics.txt` is derived from the Arphic *AR PL KaitiM GB* and
*AR PL UKai* fonts.

- Source: <https://github.com/skishore/makemeahanzi> (`graphics.txt`)
- Upstream fonts: Arphic Technology Co., Ltd. (released 1999)
- License: **Arphic Public License** (`APL`). Permissive and permits commercial
  use; it is **not** the GNU GPL.
- License text shipped at:
  `cnn_chinese_hw/data/makemeahanzi/LICENSE-Arphic-Public-License.txt`
- Derivation notice shipped at: `cnn_chinese_hw/data/makemeahanzi/COPYING`

Per the Arphic Public License, the license document is distributed verbatim with
the data, and this project's use is limited to the font-derived stroke geometry.
Only `graphics.txt` is used; `dictionary.txt` (LGPL 3) from the same project is
**not** included.

Attribution (as suggested by the project):

> Make Me a Hanzi data, © Arphic Technology Co., Ltd., used under the Arphic
> Public License. Derived from the AR PL KaitiM GB and AR PL UKai fonts.

## KanjiVG

Kanji stroke-order vector data. It is the sole licensed training corpus of the
`ccbysa` model (`license_group='ccbysa'`), and is held out **for validation
only** when training the `permissive` model — so it never enters the permissive
weights.

- Source: <https://kanjivg.tagaini.net/>
- License: Creative Commons Attribution-Share Alike 3.0
  (<https://creativecommons.org/licenses/by-sa/3.0/>)
- Copyright © Ulrich Apel / the KanjiVG project.
- License note shipped at: `cnn_chinese_hw/data/validation/kanjivg_license.txt`

Attribution (required by CC BY-SA 3.0):

> Contains information from KanjiVG (© Ulrich Apel / the KanjiVG project),
> which is made available under the Creative Commons Attribution-ShareAlike 3.0
> license. <https://kanjivg.tagaini.net/>


---

### Note on redistributing trained models

A model is a derived work of its *training* corpora (the conservative view; the
question is legally unsettled). Keep this file with any checkpoints you
redistribute so the relevant attributions travel with the weights:

- **`hw_model.permissive.pt`** — derives from Tomoe (LGPL 2.1) + Make Me a Hanzi
  (Arphic PL). No CC BY-SA. Distributable under the project's LGPL/APL terms.
- **`hw_model.ccbysa.pt`** — derives from KanjiVG (CC BY-SA 3.0); distribute it
  under CC BY-SA 3.0 with the KanjiVG attribution above.

The two licenses never meet inside a single weight file; they only meet as
combined numeric outputs at inference, which is aggregation rather than a single
co-mingled derivative. If you want to ship only commercially-clean weights,
distribute the permissive model alone — the recognizer runs whatever is present.
