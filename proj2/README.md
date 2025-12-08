# Eatsential 🥗

[![CI/CD Pipeline](https://github.com/Asoingbob225/CSC510/actions/workflows/test-coverage.yml/badge.svg)](https://github.com/Asoingbob225/CSC510/actions/workflows/test-coverage.yml)
[![Super-Linter](https://github.com/Asoingbob225/CSC510/actions/workflows/linter.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![Formatters](https://github.com/Asoingbob225/CSC510/actions/workflows/format-check.yml/badge.svg)](https://github.com/Asoingbob225/CSC510/actions/workflows/format-check.yml)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-brightgreen.svg)](https://github.com/Asoingbob225/CSC510/security/dependabot)
[![codecov](https://codecov.io/gh/Asoingbob225/CSC510/branch/main/graph/badge.svg)](https://codecov.io/gh/Asoingbob225/CSC510)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17429003-blue)](https://doi.org/10.5281/zenodo.17429003)
[![Project Status: WIP](https://img.shields.io/badge/status-wip-yellow.svg)](#)

**YOUR PLATE, YOUR RULES. PRECISION NUTRITION FOR BODY AND MIND.**

Eatsential is a LLM-powered platform that connects users to food options with unprecedented precision. By creating a holistic profile that includes allergies, fitness goals (muscle gain, weight loss, endurance), dietary preferences (vegan, keto), and even mental wellness objectives (e.g., mood-boosting foods, stress reduction), our system delivers hyper-personalized meal recommendations, recipes, and restaurant suggestions.

---

## 🎬 Demo Video

**Watch our project demonstration on YouTube:**

<div align="center">
  
[![Eatsential Demo Video](https://img.youtube.com/vi/K2KGYuvrB_Y/maxresdefault.jpg)](https://youtu.be/K2KGYuvrB_Y)

**[▶️ Watch on YouTube](https://youtu.be/K2KGYuvrB_Y)** | 2 minutes

</div>

---

## ✨ Core Features

1. **Holistic Health Tracking:** Log meals, mood, stress, and sleep to build your 360° health profile, including detailed allergens and dietary preferences.
2. **AI Recommendation Engine:** Get personalized restaurant recommendations based on your profile, goals, and strict allergen filters, with a feedback loop to learn your tastes.
3. **AI Health Concierge:** A conversational chat assistant with memory that accesses your profile to provide real-time, personalized nutrition advice.
4. **Dynamic Meal Planner:** Auto-generate 3-7 day meal plans based on your calorie/macro goals, with full support for manual swaps and adjustments.
5. **Restaurant Discovery & Community Insights:** Explore healthy restaurants via an integrated map and view community reviews focused on healthfulness and allergen safety.
6. **Secure & Managed Platform:** Features secure JWT authentication, protected routes, and a full admin dashboard for user management and system audits.

## 🚀 Quick Start Guide (2-minute onboarding)

Get Eatsential running on your local machine in just a few steps.

### Automated Setup (Recommended) ⚡

Run our one-click setup script that handles everything automatically:

```bash
# Clone the project and navigate into it
git clone <repository-url>
cd CSC510/proj2

# Run the automated setup script
./setup.sh
```

The script will:
- ✅ Check for required tools (Bun, uv, Python)
- 📦 Install all dependencies (root, frontend, backend)
- 🔐 Generate a secure JWT secret key
- 🗄️ Initialize the database with sample data
- 🎯 Set up everything you need to start coding

> [!IMPORTANT]
> **.ENV Setting Required**
>
> Please remember to set your **Gemini API keys** in the backend `.env` file to enable AI features.

**Sample Credentials:**
- Email: `admin@example.com`
- Password: `Admin123!@#`

### Manual Setup

If you prefer manual control, see our detailed [INSTALL.md](INSTALL.md) guide.

### Run the Application

Start both the frontend and backend servers simultaneously from the root directory (proj2/):

```bash
bun dev
```

The application will now be running:

- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- Backend API Docs: http://localhost:8000/docs

## 🚀 Project Status & Roadmap

This project is currently a **work in progress**. Our development is planned in two main releases:

- [x] **October Release - The Intelligent MVP**
  - [x] Week 1 — v0.1 (Auth & Profile): Established the secure foundation with user signup, email activation, JWT auth, and basic health-profile CRUD
  - [x] Week 2 — v0.2 (Health Data Management): Expanded profiles to cover detailed allergens and dietary preferences, and introduced the admin system with protected routes and user management.
  - [x] Week 3 — v0.3 (Dual-Dimension Tracking): Enabled our dual-dimension tracking: meal logging (API + UI), mental-wellness tracking (mood, stress, sleep) with goal setting, plus admin audit logs.
  - [x] Week 4 — v0.4 (AI Recommendation Engine): Shipped the AI recommendation engine, backed by a restaurant/menu database and a hybrid LLM + rules API, with strict allergen filtering and deterministic fallback to ensure safety and stability.
- **November: Release 2 - The Integrated Life Planner**
  - [x] Week 5 — v1.1 (AI Feedback Loop & Personalization): Added like/dislike feedback (API + UI) and fed signals into the recommender to adapt future results and reduce unwanted items.
  - [x] Week 6 — v1.2 (Interactive AI Health Concierge): Launched a chat-based assistant with session memory and profile access to answer nutrition questions and give real-time, personalized advice.
  - [x] Week 7 — v1.3 (Restaurant Discovery & Reviews): Introduced a map-based restaurant explorer (Mapbox/Google Maps) and community reviews for healthfulness and allergen safety.
  - [x] Week 8 — v1.4 (Dynamic AI Meal Planner): Delivered multi-day meal planning (3–7 days) that auto-generates plans from calorie/macros/goals, with manual swap and adjustment support.

## 🤖 CI/CD Pipeline

This project uses GitHub Actions for CI/CD:

- **test-coverage.yml:** Runs all tests and reports coverage for frontend and backend.
- **linter.yml:** Lints frontend and backend code for style and quality.
- **format-check.yml:** Checks code formatting for Python (Ruff) and frontend (Prettier).
- **nightly-tests.yml**: Runs all tests nightly to provide daily check for code quality

Workflow files are in `.github/workflows/`. CI status badges are at the top.

## 🤝 Contributing

We welcome contributions to Eatsential! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) to learn how you can get involved.

All contributors are expected to adhere to our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## 👥 Authors

This project is brought to you by Group 12.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

## 📚 Citation and DOI

This project is registered with Zenodo for academic citation and archival purposes. If you use this project in your research or work, please cite it using the DOI badge above.

## Need help?

Feel free to open an issue to discuss with us.
