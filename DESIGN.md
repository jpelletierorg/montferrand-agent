# Montferrand Landing Page - Creative Brief

## Brand Identity
- **Name**: Montferrand
- **Tagline direction**: "Vos appels de service continuent de rentrer, meme quand le bureau est ferme."
- **What it is**: SMS-based after-hours booking assistant for Quebec plumbers
- **What it feels like**: a capable local ally built by a guy from Longueuil who gets it
- **What it is NOT**: an AI startup, a chatbot company, a SaaS dashboard

## Visual System

### Palette
| Role | Color | Usage |
|------|-------|-------|
| Primary dark | Deep Quebec blue (`#1a3a5c`) | Headers, nav, footer, strong text |
| Primary mid | Flag blue (`#0057a8`) | Links, active states, section accents |
| Primary light | Ice blue (`#d6e8f7`) | Backgrounds, SMS bubble (agent side) |
| Background | Warm cream (`#f8f5f0`) | Page base, card backgrounds |
| White | `#ffffff` | SMS bubble (customer side), cards |
| Accent | Copper (`#c47a3a`) | CTAs, price highlights, `Commencer maintenant` |
| Text | Near-black (`#1c1c1e`) | Body copy |
| Muted | Warm grey (`#6b6b6b`) | Secondary text, captions |

### Typography
- Display/headings: a strong serif or slab with character (something like `Playfair Display`, `Bitter`, or `Zilla Slab`)
- Body: clean sans-serif (`Inter`, `Source Sans 3`, or `DM Sans`)
- SMS bubbles: monospace-adjacent or the body sans at smaller size

### Shapes and Motifs
- Circular hero frame for mascot (the "That's all folks" ring)
- Rounded SMS bubble shapes throughout
- Badge-like callout boxes for stats/pricing
- Subtle paper/poster texture on cream backgrounds
- No hard geometric SaaS patterns

### Mascot
- Jos Montferrand reimagined as a plumber
- Balanced tone: confident broad-shouldered posture, friendly face, rolled-up sleeves
- Holding or resting on a pipe wrench, maybe a tool belt
- Emerging from a circular frame in the hero
- Clean editorial illustration style, not cartoonish
- This will need to be commissioned or generated separately

### Motion
- Hero: mascot subtle reveal/slide-in, SMS conversation types out in real-time
- SMS animation: staged sequence showing a real booking flow (customer describes problem, system asks questions, confirms booking)
- Scroll-triggered section reveals (subtle fade-up)
- CTA pulse on the sticky button
- No confetti, no parallax, no decorative motion

## Page Structure

### 1. Nav
- Logo (Montferrand wordmark or small mascot mark)
- Minimal links: `Comment ca marche` | `Prix` | `A propos` | `Contact`
- CTA button: `Commencer maintenant`

### 2. Hero
- Left: mascot in circular frame
- Right:
  - Headline: `Vos appels de service continuent de rentrer, meme quand le bureau est ferme.`
  - Subhead: `Montferrand repond a vos clients par texto, pose les bonnes questions de plombier et vous envoie un resume pret a traiter le matin.`
  - Primary CTA: `Essayer la demo` (copper button, opens demo modal)
  - Secondary CTA: `Commencer maintenant` (outlined, scrolls to onboarding section)
- The SMS animation plays on loop or on a short delay, showing a condensed booking exchange

### 3. Outcome Strip
- `Repond a vos clients 24/7 par texto`
- `Pose les bonnes questions de plomberie`
- `Vous envoie les rendez-vous prets a traiter`

### 4. Comment ca marche
- 3-step visual flow:
  1. `Votre client texte` - customer describes their problem by SMS
  2. `Montferrand qualifie` - the system asks diagnostic questions, collects name/address/details
  3. `Vous recevez le resume` - plumber gets a clean booking summary by SMS, email, or Google Calendar at the start of the workday

### 5. Essayer la demo
- Prominent section near top-third of page
- CTA: `Essayer la demo maintenant`
- Click opens modal:
  - Friendly plain-language explanation:
    - "Ceci est un environnement de demonstration."
    - "Les prix, les heures et les disponibilites sont fictifs."
    - "Le but est de vous montrer exactement comment vos clients vont interagir avec le systeme."
    - "Vous pouvez aussi demander une installation directement dans la conversation."
  - Checkbox: `J'ai compris, montrez-moi le numero`
  - On check: reveals the SMS number with a `Texter maintenant` link (opens native SMS on mobile)

### 6. Prix et ROI
- Headline: `Essayez sans risque pendant 30 jours. 100 % satisfait ou rembourse.`
- Pricing card:
  - `30 jours gratuits`
  - `59,99 $ / mois ensuite`
  - `Aucun contrat a long terme`
- ROI examples (static cards or small table):
  - `1 appel de service de plus par mois` -> le systeme se paie tout seul
  - `3 appels de service de plus` -> X $ de profit supplementaire
  - `5 appels de service de plus` -> X $ de profit supplementaire
- Note: we need an average job value assumption to fill in the X. Start with `250 $` as a conservative residential service call, then adjust.

### 7. Commencer maintenant / Onboarding
- Headline: `Pret a essayer? On s'occupe de tout.`
- Simple step flow:
  1. `On vous appelle` - 10 minutes, on pose des questions simples sur votre entreprise
  2. `On configure votre assistant` - vos heures, vos services, votre facon de parler
  3. `On gere le paiement par virement Interac` - simple et direct
  4. `Votre assistant est en ligne` - vos clients peuvent texter des ce soir
- CTA: `Commencer maintenant` (copper button, links to contact form or reveals phone/email)

### 8. Bientot disponible
- `Assistant vocal - bientot disponible`
- Brief one-liner: `Vos clients pourront aussi appeler et parler directement a votre assistant Montferrand.`

### 9. A propos
- Your photo (ideally the Longueuil Beach t-shirt)
- Headline: `Fait a Longueuil Beach`
- Copy direction: `Je m'appelle Jonathan. J'ai bati Montferrand parce que je crois que les plombiers du Quebec meritent les memes outils que les grandes entreprises. C'est pas complique: vous faites de la plomberie, je m'occupe de la techno.`
- Keep it honest, short, human. No startup jargon.

### 10. Footer
- Logo
- Links: `Contact` | `Confidentialite` | `Conditions`
- `Fait au Quebec`

### Sticky Mobile CTA
- Fixed bottom bar on mobile: `Essayer la demo`

## Additional Pages

### Contact page
- Simple form: name, business name, phone, email, short message
- Or phone number plus email displayed directly
- Same nav/footer as landing

### Privacy / Legal
- Standard privacy policy
- Minimal, plain French

## Technical Plan
- Static HTML/CSS/JS on Cloudflare Pages
- CSS animations for the SMS sequence (no heavy JS framework needed)
- Intersection Observer for scroll-triggered reveals
- Google Analytics 4 plus Google Ads conversion tracking
- Mobile-first responsive design
- Separate `/contact` and `/confidentialite` routes

## Remaining Inputs Needed
1. Average service call value - to fill in the ROI examples (start with `250 $` by default)
2. Demo phone number
3. Mascot illustration
4. Founder photo
5. Business contact info
