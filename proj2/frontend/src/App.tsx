import { Routes, Route } from 'react-router';
import Welcome from './pages/Welcome';
import Signup from './pages/Signup';
import VerifyEmail from './pages/VerifyEmail';
import Dashboard from './pages/Dashboard';
import DailyLog from './pages/DailyLog';
import Login from './pages/Login';
import NotFound from './pages/NotFound';
import HealthProfileWizard from './pages/HealthProfileWizard';
import HealthProfilePage from './pages/HealthProfile';
import WellnessTracking from './pages/WellnessTracking';
import AdminRoute from './components/AdminRoute';
import AdminLayout from './components/AdminLayout';
import AdminDashboard from './pages/admin/AdminDashboard';
import UserManagement from './pages/admin/UserManagement';
import AllergenManagement from './pages/admin/AllergenManagement';
import AdminSettings from './pages/admin/AdminSettings';
import AuditDashboard from './pages/admin/AuditDashboard';
import { ChatWidget } from './components/chat/ChatWidget';

function App() {
  return (
    <>
      <Routes>
        <Route path="/" element={<Welcome />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/verify-email" element={<VerifyEmail />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/daily-log" element={<DailyLog />} />
        <Route path="/health-profile-wizard" element={<HealthProfileWizard />} />
        <Route path="/health-profile" element={<HealthProfilePage />} />
        <Route path="/wellness-tracking" element={<WellnessTracking />} />

        {/* Admin routes - protected */}
        <Route
          path="/system-manage"
          element={
            <AdminRoute>
              <AdminLayout />
            </AdminRoute>
          }
        >
          <Route index element={<AdminDashboard />} />
          <Route path="users" element={<UserManagement />} />
          <Route path="allergens" element={<AllergenManagement />} />
          <Route path="audit" element={<AuditDashboard />} />
          <Route path="settings" element={<AdminSettings />} />
        </Route>

        <Route path="*" element={<NotFound />} />
      </Routes>
      <ChatWidget />
    </>
  );
}

export default App;
