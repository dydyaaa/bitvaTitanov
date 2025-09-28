import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { isAuthenticated } from "./services/auth";
import LoginPage from "./pages/LoginPage/LoginPage";
import RegisterPage from "./pages/RegisterPage/RegisterPage";
import ChatPage from "./pages/ChatPage/ChatPage";
import DocumentsPage from "./pages/DocumentsPage/DocumentsPage";
import StatsPage from "./pages/StatsPage/StatsPage";

export default function App() {
  const AuthedRoute = ({ children }) => {
    return isAuthenticated() ? children : <Navigate to="/login" />;
  };

  const GuestRoute = ({ children }) => {
    return isAuthenticated() ? <Navigate to="/chat" /> : children;
  };

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to={isAuthenticated() ? "/chat" : "/login"} />} />

        <Route path="/login" element={<GuestRoute><LoginPage /></GuestRoute>} />
        <Route path="/register" element={<GuestRoute><RegisterPage /></GuestRoute>} />
        <Route path="/chat" element={<AuthedRoute><ChatPage /></AuthedRoute>} />
        <Route path="/documents" element={<AuthedRoute><DocumentsPage /></AuthedRoute>} />
        <Route path="/stats" element={<AuthedRoute><StatsPage /></AuthedRoute>} />

        <Route path="*" element={<h1>404 | Страница не найдена</h1>} />
      </Routes>
    </BrowserRouter>
  );
}
