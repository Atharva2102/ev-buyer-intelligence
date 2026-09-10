"use client";

import Image from "next/image";
import Link from "next/link";
import { motion, useReducedMotion } from "motion/react";
import {
  ArrowRight, BatteryCharging, BrainCircuit, ChartNoAxesCombined, CloudCog,
  Github, Linkedin, ScanSearch, ShieldCheck, SlidersHorizontal, UsersRound
} from "lucide-react";
import { publicAsset, siteLinks } from "../lib/site";

const workflow = [
  { number: "01", title: "Prepare the data", text: "Python ETL validates source files, engineers useful features, and builds reusable analytics tables in DuckDB." },
  { number: "02", title: "Train and compare", text: "Leak-free cross-validation compares linear, tree, ensemble, and neural tabular models using ROC AUC." },
  { number: "03", title: "Serve predictions", text: "FastAPI loads a trained LightGBM artifact and records each interactive score in SQLite or AWS DynamoDB." },
  { number: "04", title: "Communicate decisions", text: "The dashboard turns model outputs into segment, operations, quality, and scenario views for non-technical stakeholders." }
];

const dashboardRoutes = [
  { href: "/dashboard", icon: ChartNoAxesCombined, title: "Decision overview", text: "Summarizes the scoring population and compares which modeled interventions affect the most buyers." },
  { href: "/live-scoring", icon: CloudCog, title: "Scoring operations", text: "Monitors real API requests, inference speed, failures, event sources, and prediction bands." },
  { href: "/segments", icon: UsersRound, title: "Buyer segments", text: "Finds groups with above-average predicted intent without pretending the model proves why they behave that way." },
  { href: "/market-dna", icon: ScanSearch, title: "Market comparison", text: "Shows where the competition population differs from the original 10,000-row reference dataset." },
  { href: "/simulator", icon: SlidersHorizontal, title: "Profile simulator", text: "Lets a user change one hypothetical buyer profile and request a new model probability." },
  { href: "/model", icon: BrainCircuit, title: "Model development", text: "Explains validation, experiment progression, the winning TabM submission, and the separate serving model." }
];

const audiences = [
  { title: "Growth and market teams", text: "Use segment rankings and buyer profiles to decide which audiences deserve deeper research or outreach." },
  { title: "Policy and infrastructure teams", text: "Compare directional subsidy, charging-access, and education scenarios before commissioning a causal study." },
  { title: "Data and ML teams", text: "Review model evidence, data drift, quality checks, cloud architecture, and live inference operations in one product." }
];

export default function LandingPage() {
  const reduce = useReducedMotion();
  const reveal = (delay = 0) => ({
    initial: reduce ? false as const : { opacity: 0, y: 24 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true, amount: 0.2 },
    transition: { duration: 0.65, delay, ease: [0.16, 1, 0.3, 1] as [number, number, number, number] }
  });

  return (
    <main className="landing-root">
      <a className="skip-link" href="#landing-main">Skip to content</a>
      <section className="landing-hero" id="landing-main">
        <motion.div className="hero-media" initial={reduce ? false : { scale: 1.04 }} animate={{ scale: 1 }} transition={{ duration: 1.8, ease: [0.16, 1, 0.3, 1] }}>
          <Image src={publicAsset("/images/ev-hero.png")} alt="Unbranded electric sedan at a modern charging facility" fill priority sizes="100vw" />
        </motion.div>
        <div className="hero-shade" />
        <div className="hero-grid" aria-hidden="true" />
        <header className="landing-nav">
          <Link href="/" className="landing-brand"><span><BatteryCharging size={22} /></span><strong>EV Buyer Intelligence</strong></Link>
          <nav aria-label="Main navigation"><Link className="active" href="/">Home</Link><Link href="#features">Features</Link><Link className="nav-cta" href="/dashboard">Dashboard</Link></nav>
        </header>
        <div className="hero-content">
          <span className="hero-kicker">End-to-end data science portfolio</span>
          <h1>EV Buyer<br />Intelligence</h1>
          <p>A working decision-support platform that turns synthetic EV adoption data into validated predictions, stakeholder-ready analysis, and monitored cloud scoring.</p>
          <div className="hero-actions"><Link href="/dashboard">Explore the dashboard <ArrowRight size={18} /></Link><Link href="#project">Understand the project</Link></div>
        </div>
        <div className="hero-proof" aria-label="Project highlights">
          <div><strong>668,665</strong><span>training profiles</span></div>
          <div><strong>0.94480</strong><span>best public ROC AUC</span></div>
          <div><strong>AWS</strong><span>cloud event pipeline</span></div>
        </div>
        <aside className="social-rail" aria-label="Social links"><span>Connect</span><i /><a href={siteLinks.github} target="_blank" rel="noreferrer" aria-label="GitHub"><Github size={21} /></a><a href={siteLinks.linkedin} target="_blank" rel="noreferrer" aria-label="LinkedIn"><Linkedin size={21} /></a></aside>
      </section>

      <section id="project" className="project-purpose">
        <motion.div className="purpose-heading" {...reveal()}><span className="landing-label">The project brief</span><h2>Built to emulate the work around a model, not only the model itself.</h2></motion.div>
        <motion.div className="purpose-copy" {...reveal(0.08)}><p>Imagine an EV strategy team asking: who appears most open to buying, which access barriers deserve investigation, how reliable is the data, and is the scoring service working? This platform creates one place to answer those questions.</p><p>It demonstrates data engineering, model evaluation, API development, cloud event storage, dashboard design, and the stakeholder communication needed to connect them.</p></motion.div>
      </section>

      <section id="features" className="landing-features">
        <div className="features-intro"><span>How it works</span><h2>From source files<br />to a decision surface.</h2><p>Each layer produces evidence for the next. Nothing on the dashboard requires a stakeholder to read a notebook first.</p></div>
        <div className="workflow-grid">
          {workflow.map((item, index) => <motion.article key={item.number} {...reveal(index * 0.06)}><span>{item.number}</span><div><h3>{item.title}</h3><p>{item.text}</p></div></motion.article>)}
        </div>
      </section>

      <section className="dashboard-tour">
        <motion.div className="tour-heading" {...reveal()}><span className="landing-label">Dashboard guide</span><h2>Six views, each built around a stakeholder question.</h2><p>Open any view below. Every page explains what the numbers mean, how to read them, and what not to infer.</p></motion.div>
        <div className="tour-grid">
          {dashboardRoutes.map((item, index) => <motion.div key={item.href} {...reveal(index * 0.045)}><Link href={item.href}><item.icon /><span>0{index + 1}</span><h3>{item.title}</h3><p>{item.text}</p><b>Open view <ArrowRight size={15} /></b></Link></motion.div>)}
        </div>
      </section>

      <section className="audience-section">
        <div className="audience-intro"><span className="landing-label">Who benefits</span><h2>One product, different decisions.</h2><p>The interface separates strategic questions from technical evidence while keeping both available.</p></div>
        <div className="audience-list">{audiences.map((item, index) => <motion.article key={item.title} {...reveal(index * .07)}><span>0{index + 1}</span><div><h3>{item.title}</h3><p>{item.text}</p></div></motion.article>)}</div>
      </section>

      <section className="evidence-note">
        <ShieldCheck />
        <div><span className="landing-label">Read the evidence honestly</span><h2>A portfolio demonstration, not a purchasing oracle.</h2><p>The competition data is synthetic and contains vehicle classes, not automaker brands. Model probabilities describe patterns in this dataset. Policy scenarios are directional assumptions, not causal estimates, and should support investigation rather than automated high-impact decisions.</p></div>
        <Link href="/model">Review model evidence <ArrowRight size={16} /></Link>
      </section>

      <section className="landing-cta"><div><span>See the system working</span><h2>Start with the decision overview.</h2></div><Link href="/dashboard">Launch dashboard <ArrowRight size={18} /></Link></section>
      <footer className="landing-footer"><span>EV Buyer Intelligence</span><span>Data engineering, ML, cloud, and product analytics</span><div><a href={siteLinks.github}>GitHub</a><a href={siteLinks.linkedin}>LinkedIn</a></div></footer>
    </main>
  );
}
