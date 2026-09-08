"use client";

import Image from "next/image";
import Link from "next/link";
import { motion, useReducedMotion } from "motion/react";
import { ArrowRight, BatteryCharging, DatabaseZap, Github, Linkedin, ScanSearch, Sparkles } from "lucide-react";
import { siteLinks } from "../lib/site";

const features = [
  { icon: DatabaseZap, title: "Warehouse analytics", text: "A reproducible ETL pipeline turns raw survey and competition data into trusted decision marts.", stat: "286,571", label: "profiles scored" },
  { icon: ScanSearch, title: "Buyer intelligence", text: "Explore intent by vehicle type, market context, charging access, and adoption barriers.", stat: "58", label: "market segments" },
  { icon: Sparkles, title: "Scenario planning", text: "Test buyer profiles and compare policy interventions against the current population.", stat: "4", label: "policy levers" }
];

export default function LandingPage() {
  const reduce = useReducedMotion();
  return (
    <main className="landing-root">
      <a className="skip-link" href="#landing-main">Skip to content</a>
      <section className="landing-hero" id="landing-main">
        <motion.div className="hero-media" initial={reduce ? false : { scale: 1.035 }} animate={{ scale: 1 }} transition={{ duration: 1.6, ease: [0.16, 1, 0.3, 1] }}>
          <Image src="/images/ev-hero.png" alt="Unbranded electric sedan at a modern charging facility" fill priority sizes="100vw" />
        </motion.div>
        <div className="hero-shade" />
        <header className="landing-nav">
          <Link href="/home" className="landing-brand"><span><BatteryCharging size={22} /></span><strong>EV Buyer Intelligence</strong></Link>
          <nav aria-label="Main navigation"><Link className="active" href="/home">Home</Link><Link href="#features">Features</Link><Link className="nav-cta" href="/">Dashboard</Link></nav>
        </header>
        <div className="hero-content">
          <span className="hero-kicker">From raw data to buyer decisions</span>
          <h1>EV Buyer<br />Intelligence</h1>
          <p>A working analytics platform for scoring purchase intent, finding market opportunities, and testing adoption strategies.</p>
          <div className="hero-actions"><Link href="/">Open dashboard <ArrowRight size={18} /></Link><Link href="#features">Explore features</Link></div>
        </div>
        <aside className="social-rail" aria-label="Social links"><span>Connect</span><i /><a href={siteLinks.github} target="_blank" rel="noreferrer" aria-label="GitHub"><Github size={21} /></a><a href={siteLinks.linkedin} target="_blank" rel="noreferrer" aria-label="LinkedIn"><Linkedin size={21} /></a></aside>
      </section>

      <section id="features" className="landing-features">
        <div className="features-intro"><span>Inside the platform</span><h2>One data product.<br />Three working layers.</h2><p>The interface is backed by a real warehouse, analytics API, and machine-learning score pipeline.</p></div>
        <div className="feature-grid">
          {features.map((feature, index) => <motion.article key={feature.title} initial={reduce ? false : { opacity: 0, y: 28 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.25 }} transition={{ duration: 0.6, delay: index * 0.08 }}><feature.icon size={25} strokeWidth={1.6} /><div><h3>{feature.title}</h3><p>{feature.text}</p></div><footer><strong>{feature.stat}</strong><span>{feature.label}</span></footer></motion.article>)}
        </div>
        <div className="landing-cta"><div><span>See the system working</span><h2>Turn signals into an adoption strategy.</h2></div><Link href="/">Launch dashboard <ArrowRight size={18} /></Link></div>
      </section>
      <footer className="landing-footer"><span>EV Buyer Intelligence</span><span>Data engineering, ML, and product analytics</span><div><a href={siteLinks.github}>GitHub</a><a href={siteLinks.linkedin}>LinkedIn</a></div></footer>
    </main>
  );
}
