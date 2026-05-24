import { Route, Routes } from "react-router-dom"

import Analyzing from "@/screens/Analyzing"
import Coaching from "@/screens/Coaching"
import Home from "@/screens/Home"
import Report from "@/screens/Report"
import Setup from "@/screens/Setup"

function App() {
  return (
    <div
      className="min-h-screen font-sans text-[color:var(--color-aura-ink)]"
      style={{
        backgroundImage: "var(--gradient-bg-soft)",
        paddingTop: "env(safe-area-inset-top)",
      }}
    >
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/setup" element={<Setup />} />
        <Route path="/coaching" element={<Coaching />} />
        <Route path="/analyzing" element={<Analyzing />} />
        <Route path="/report" element={<Report />} />
      </Routes>
    </div>
  )
}

export default App
