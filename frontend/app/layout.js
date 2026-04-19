import './globals.css'
export const metadata = { title: 'Custom GPT', description: 'Your own AI — no API keys' }
export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
