import { Route, Routes } from "react-router-dom"

import Coaching from "@/screens/Coaching"
import Countdown from "@/screens/Countdown"
import ExerciseDetail from "@/screens/ExerciseDetail"
import ExpertVideo from "@/screens/ExpertVideo"
import Feedback from "@/screens/Feedback"
import Home from "@/screens/Home"
import Setup from "@/screens/Setup"
import Splash from "@/screens/Splash"

function App() {
  return (
    <div
      className="min-h-screen font-sans text-white"
      style={{
        backgroundImage: "var(--gradient-bg-soft)",
        backgroundAttachment: "fixed",
        paddingTop: "env(safe-area-inset-top)",
      }}
    >
      <Routes>
        <Route path="/" element={<Splash />} />
        <Route path="/home" element={<Home />} />
        <Route path="/exercise" element={<ExerciseDetail />} />
        <Route path="/setup" element={<Setup />} />
        <Route path="/countdown" element={<Countdown />} />
        <Route path="/coaching" element={<Coaching />} />
        <Route path="/feedback" element={<Feedback />} />
        <Route path="/expert/:exerciseId" element={<ExpertVideo />} />
      </Routes>
    </div>
  )
}

export default App
