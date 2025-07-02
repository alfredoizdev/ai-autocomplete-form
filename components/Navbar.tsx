import Link from "next/link";

const Navbar = () => {
  return (
    <div className="flex items-center justify-center p-4 bg-gray-900 text-white">
      <ul className="flex items-center space-x-4">
        <li>
          <Link href="/" className="text-md font-semibold">
            AI Bio Autocomplete
          </Link>
        </li>
        <li className="ml-6">
          <Link href="/ai-image" className="text-md font-semibold">
            AI Image Analysis
          </Link>
        </li>
      </ul>
    </div>
  );
};

export default Navbar;
