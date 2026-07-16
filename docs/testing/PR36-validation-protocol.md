# Protocole de validation — PR #36 (asks LipDub app : B1/B2/B4/B6)

Valide les quatre changements de la PR #36 sur du matériel réel, du plus rapide au plus long.
Chaque section donne : préparation, commande exacte, critère **PASS/FAIL** observable.

**Prérequis communs**

```bash
git checkout feature/lipdub-app-asks
# Binaire Release (le binaire `swift build` crashe sur le metallib MLX) :
xcodebuild -scheme ltx-video -configuration Release -derivedDataPath .xcodebuild \
  -destination 'platform=macOS' -skipPackagePluginValidation -skipMacroValidation build
cd .xcodebuild/Build/Products/Release   # OBLIGATOIRE : metallib à côté du binaire
```

Assets utilisés (déjà présents) :
- Vidéo de référence : `docs/examples/lipdub/lipdub-teaser-french-ours-768x512-121f.mp4` (5 s, parole FR)
- IC-LoRA LipDub : `~/Pictures/FluxforgeStudio/Models/ltx-lora-lipdub/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors`

Notation ci-dessous : `$REPO` = racine du repo, `$REF` = la vidéo de référence, `$LORA` = le LoRA.

---

## Étape 0 — Suites unitaires (2 min)

Couvre déjà : cas plancher -32,5 dB (B2), garde bruit-faible (B2), bornes 481/489 (B1).

```bash
cd $REPO
swift test --filter "AudioPreprocessor|LTXConfig"
```

**PASS** : 49 tests verts (11 AudioPreprocessor + 38 LTXConfig).

---

## Étape 1 — B4 : plus de « missing » au chargement (≈ 3 min, aucun GPU-long)

Tout run chargeant le transformer audio suffit ; le plus court est un lipdub qu'on
n'a même pas besoin de laisser finir.

```bash
./ltx-video lipdub 'A person speaking in French saying: "Bonjour à tous."' \
  --reference-video $REF --lora $LORA \
  -w 384 -h 256 -f 33 --debug -o /tmp/b4.mp4 2>&1 | tee /tmp/b4.log
# (interruptible Ctrl-C dès que le chargement des poids est passé)
```

**PASS** :
- `grep "affine-free block norms" /tmp/b4.log` → une ligne debug
  `Transformer: 384 affine-free block norms left at default weight=1 (expected)`
- `grep "\[Weights\].*Missing" /tmp/b4.log` → **rien** (avant la PR : `384 missing` + liste `audio_norm1…`)
- ⚠️ Si une ligne `Missing:` apparaît quand même avec d'AUTRES clés → vrai trou de mapping, STOP et ouvrir une issue.

**FAIL** si la ligne `[Weights] … missing` mentionne encore les normes de bloc.

## Étape 1b — B6 (log) : libellé de fusion

Dans le même `/tmp/b4.log` :

**PASS** : `[lipdub] LoRA fused: 1344 / 1344 layer-pairs (100.0%)` (avant : `4032 / 1344 layers (300.0%)`).

---

## Étape 2 — B2 : fenêtre de parole sur voix au plancher haut (≈ 10 min)

### 2a. Fabriquer un audio « voix enrollée » reproductible

On prend la piste du teaser (vraie parole + vrais silences) et on lui injecte un
plancher de bruit à ≈ -32,5 dB — la signature mesurée des voix enrollées :

```bash
ffmpeg -y -i $REF -vn -ac 1 -ar 16000 /tmp/target_clean.wav
ffmpeg -y -i /tmp/target_clean.wav -filter_complex \
  "anoisesrc=color=pink:amplitude=0.024:duration=10[n];[0][n]amix=inputs=2:duration=first:normalize=0" \
  -ar 16000 -ac 1 /tmp/target_enrolled.wav
# Contrôle : le plancher doit être au-dessus de -35 dB
ffmpeg -i /tmp/target_enrolled.wav -af silencedetect=n=-35dB -f null - 2>&1 | grep silence_ || echo "OK: aucun silence détecté à -35 dB (cas pathologique reproduit)"
```

### 2b. Run LipDub avec cet audio cible

```bash
./ltx-video lipdub 'A person speaking in French saying: "Bonjour à tous."' \
  --reference-video $REF --target-audio /tmp/target_enrolled.wav --lora $LORA \
  -w 384 -h 256 -f 121 --debug -o /tmp/b2.mp4 2>&1 | tee /tmp/b2.log
```

**PASS** :
- Le log d'alignement (`grep -i "align\|window\|stretch\|rate" /tmp/b2.log`) montre des
  fenêtres source/cible **strictement plus étroites que le clip entier** et un rate cohérent (≈ 1.0 ici,
  puisque la cible EST la source bruitée) — avant la PR : fenêtre = clip entier ou erreur
  `Speech window detection failed`.
- Contrôle visuel de `/tmp/b2.mp4` : bouche synchrone, pas de décalage d'attaque.
- Contre-épreuve (non-régression, TTS propre) : relancer 2b avec `/tmp/target_clean.wav` → même comportement qu'avant la PR.

**FAIL** : fenêtre = clip entier avec l'audio bruité, ou régression sur l'audio propre.

---

## Étape 3 — B1 : 481 frames (≈ 30-60 min, le plus long)

### 3a. Validation mécanique (instantané)

```bash
./ltx-video generate "test" -w 384 -h 256 -f 489 -o /tmp/nope.mp4 ; echo "exit=$?"
```

**PASS** : erreur `Number of frames must be between 9 and 481 (20 s at 24 fps — the RoPE positional range…)`, pas de crash.

### 3b. Run long réel — 481 frames = 20 s

```bash
./ltx-video generate \
  "a slow cinematic drone shot over a coastline at sunset, waves rolling in" \
  -w 512 -h 512 -f 481 --seed 42 -o /tmp/b1_481.mp4 --debug 2>&1 | tee /tmp/b1.log
```

**PASS** :
- Run complet sans erreur ; `ffprobe /tmp/b1_481.mp4` → 481 frames, ~20,04 s à 24 fps.
- Pas d'explosion mémoire (surveiller avec SiliconScope + `--beacon` 😉).
- **Qualité** (le vrai juge) : comparer visuellement à un run 241 frames même seed/prompt
  (`-f 241 -o /tmp/b1_241.mp4`). Attendu : cohérence temporelle maintenue sur 20 s ;
  un léger ramollissement au-delà de ~10 s est toléré (hors distribution d'entraînement),
  une dégénérescence (boucles, flou massif, artefacts géométriques) ne l'est pas.
- Idéalement : un LipDub 481f avec réplique de ~19 s (le cas d'usage réel du doublage).

**FAIL** : crash, OOM, ou dégénérescence visuelle franche entre 241 et 481 → re-plafonner
(le rationale RoPE reste valable mais la distribution d'entraînement gagnerait).

---

## Étape 4 — B6 : réutilisation de la fusion (test E2E gated, ≈ 15-25 min)

**Non testable via CLI** (un process par run → rechargement de toute façon).
Un test E2E dédié pilote deux `generateLipDub` consécutifs dans le même process :
`Tests/LTXVideoTests/LipDubReuseE2ETests.swift`, gated derrière `LTX_E2E_LIPDUB=1`.

```bash
cd $REPO
TEST_RUNNER_LTX_E2E_LIPDUB=1 \
TEST_RUNNER_LTX_E2E_LIPDUB_LORA=$LORA \
xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
  -derivedDataPath .xcodebuild-tests-rel -skipPackagePluginValidation -skipMacroValidation \
  -configuration Release ENABLE_TESTABILITY=YES test \
  -only-testing:LTXVideoTests/LipDubReuseE2ETests
```

Deux pièges de cette commande :
- `ENABLE_TESTABILITY=YES` est **obligatoire** en Release (`@testable import` n'est activé
  par défaut qu'en Debug — sans lui : `unable to resolve Swift module dependency to a
  compatible module: 'LTXVideo'`).
- derivedDataPath **dédié** (`.xcodebuild-tests-rel`) : ne pas réutiliser celui des runs
  Debug, les modules des deux configurations ne sont pas compatibles.
- Les variables d'environnement doivent porter le préfixe `TEST_RUNNER_` pour atteindre
  le process de test.

Le test vérifie la machine à états complète :
1. `fusedLipDubLoRAPath == nil` après chargement ;
2. run 1 → fusion, état = chemin du LoRA, transformer survivant (`.disabled`) ;
3. run 2 même LoRA → **réutilisation** (log `[lipdub] LoRA already fused (same file)`),
   run complet, état inchangé — avant la PR : double fusion = sortie corrompue ;
4. `generateVideo` et `generateRetake` pendant la fusion → `LTXError` (avant : corruption silencieuse) ;
5. LoRA différent sans rechargement → `LTXError` explicite.

**PASS** : `TEST SUCCEEDED` + dans le log, run 2 nettement plus court que run 1
(pas de re-fusion) et la ligne `reusing fused transformer`.

**Validation croisée qualité (recommandée)** : exporter la frame 0 des deux runs —
elles doivent être identiques (même seed) ou visuellement saines ; une sortie
« brûlée »/saturée au run 2 signerait une double fusion.

### 4b. Dans l'app Fluxforge (validation finale du gain)

Avec le framework de cette branche : pipeline en `MemoryOptimizationConfig.disabled`,
segmenter un dialogue en 3 segments, chronométrer :

**PASS** : segments 2 et 3 sans temps de rechargement du 22B (gain attendu : ~1-8 min/segment
selon disque/quantization) ; `pipeline.fusedLipDubLoRAPath` non-nil entre les segments.

---

## Étape 5 — Non-régression générale (déjà exécutée sur la branche, à refaire si retouches)

```bash
xcodebuild -scheme ltx-video-swift-mlx-Package -destination 'platform=macOS' \
  -derivedDataPath .xcodebuild-tests -skipPackagePluginValidation -skipMacroValidation test
```

**PASS** : `** TEST SUCCEEDED **`.

Plus un run de référence standard (aucun changement attendu) :

```bash
./ltx-video generate "a red ball bouncing on a wooden table" -w 512 -h 512 -f 121 --seed 42 -o /tmp/reg.mp4
```

---

## Récapitulatif

| # | Cible | Durée | Automatisé | Critère principal |
|---|-------|-------|------------|-------------------|
| 0 | B1+B2 unitaires | 2 min | ✅ `swift test` | 49 verts |
| 1 | B4 + log B6 | 3 min | grep | plus de `Missing:`, `1344/1344 (100.0%)` |
| 2 | B2 réel | 10 min | grep + visuel | fenêtres détectées malgré plancher -32,5 dB |
| 3 | B1 réel | 30-60 min | ffprobe + visuel | 481f OK, qualité comparable à 241f |
| 4 | B6 E2E | 15-25 min | ✅ test gated | TEST SUCCEEDED, run 2 sans fusion |
| 5 | Non-régression | 20 min | ✅ xcodebuild | TEST SUCCEEDED |
