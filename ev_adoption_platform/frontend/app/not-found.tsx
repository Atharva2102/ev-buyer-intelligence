import Link from "next/link";
import { ArrowLeft, BatteryCharging } from "lucide-react";

export default function NotFound() {
  return <main className="not-found"><BatteryCharging size={38} /><span>404</span><h1>This route is off the map.</h1><p>Return to the analytics workspace or visit the project home.</p><div><Link href="/"><ArrowLeft size={17} /> Dashboard</Link><Link href="/home">Project home</Link></div></main>;
}
