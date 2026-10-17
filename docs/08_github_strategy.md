# 8. GitHub Presentation & Portfolio Strategy

---

## 8.1 Key Elements for a Top 1% Portfolio Repository

When technical recruiters, engineering managers, and ML leads review GitHub repositories, they look for specific proof signals:

1. **Working Live Demo (Zero Barrier to Entry):**
   * Reviewers rarely clone repositories or build environments on their own machines.
   * By providing a live interactive web application hosted on GitHub Pages ([`https://rohitsingh83.github.io/AI-Predictive-Maintenance-IoT/`](https://rohitsingh83.github.io/AI-Predictive-Maintenance-IoT/)), anyone can interact with the digital twin, sliders, and work order generator in 5 seconds on mobile or desktop.

2. **Visual Proof in README:**
   * High-resolution screenshots placed directly in the `README.md` demonstrate that the project actually works and produces real outputs.
   * Tables with concrete numbers (ROC-AUC 0.978, 54 min notice, $1.19M savings) rather than vague claims.

3. **Software Engineering Rigor:**
   * Modular `src/` layout separating config, simulation, features, training, and alerting.
   * Automated test suite (`pytest`) integrated with GitHub Actions CI.
   * `requirements.txt`, `Dockerfile`, `docker-compose.yml`, and `Makefile`.

---

## 8.2 GitHub Pages Configuration Steps

To enable the live interactive web demo on your GitHub account:

1. Push the repository to GitHub:
   ```bash
   git push -u origin main
   ```
2. Open your repository on GitHub in a web browser.
3. Click on **Settings** (top right tab).
4. Click on **Pages** (in the left navigation under "Code and automation").
5. Under **Build and deployment → Branch**:
   * Select branch: `main`
   * Select folder: `/docs`
   * Click **Save**.
6. GitHub Pages will build and deploy within ~60 seconds to:
   ```
   https://rohitsingh83.github.io/AI-Predictive-Maintenance-IoT/
   ```
