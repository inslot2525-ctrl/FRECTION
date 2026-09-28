import { lazy, Suspense, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import LandingPage from './LandingPage'

// The dashboard (and its graph library) loads on demand, so the landing page opens faster
const loadDashboard = () => import('./Dashboard')
const Dashboard = lazy(loadDashboard)

export default function App() {
  // Prefetch the dashboard in the background once the landing page is idle
  useEffect(() => {
    const t = setTimeout(loadDashboard, 1500)
    return () => clearTimeout(t)
  }, [])

  return (
    <BrowserRouter>
      <Suspense fallback={<div className="min-h-screen bg-black" />}>
        <Routes>
          <Route path="/"        element={<LandingPage />} />
          <Route path="/detect"  element={<Dashboard />} />
          {/* catch-all → home */}
          <Route path="*"        element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  )
}
