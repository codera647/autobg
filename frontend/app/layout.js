import "./globals.css";

export const metadata = {
  title: "AutoBG — Car Background Studio",
  description: "Upload a car, pick a studio background, get a premium composite.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
